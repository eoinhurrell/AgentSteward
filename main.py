"""
Improved Agent Steward with SQLite Memory using Claude
-----------------------------------------------------
This version uses ChatAnthropic with Claude for improved reasoning
capabilities and better handling of memory operations.
"""

import json
import logging
import os
import sqlite3
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from langchain_anthropic import ChatAnthropic

# LangChain and LangGraph imports
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# For generating unique IDs
from nanoid import generate as nanoid_generate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("agent_steward.log")],
)
logger = logging.getLogger("agent_steward")

# Configuration
CONFIG = {
    "model": "claude-3-sonnet-20240229",  # Claude model to use
    "db_path": "agent_steward.db",  # SQLite database path
    "debug": True,  # Enable debug logging
    "connection_timeout": 30,  # SQLite connection timeout
    "temperature": 0.2,  # Lower temperature for more predictable responses
}


# Helper Functions
def get_formatted_date(date_str: Optional[str] = None) -> str:
    """Convert a date string to ISO format or return today's date if None."""
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")

    try:
        # Handle relative dates
        today = datetime.now()
        if date_str.lower() == "today":
            return today.strftime("%Y-%m-%d")
        elif date_str.lower() == "tomorrow":
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        elif date_str.lower() == "yesterday":
            return (today - timedelta(days=1)).strftime("%Y-%m-%d")

        # Try various date formats
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y"]:
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue

        # Default to parsing as ISO format
        return datetime.fromisoformat(date_str).strftime("%Y-%m-%d")
    except Exception as e:
        logger.warning(
            f"Failed to parse date '{date_str}', using today's date: {str(e)}"
        )
        return datetime.now().strftime("%Y-%m-%d")


def normalize_tags(tags: Union[str, List[str]]) -> str:
    """Normalize tags to a comma-separated string."""
    if isinstance(tags, list):
        return ",".join([str(tag).strip() for tag in tags])
    elif isinstance(tags, str):
        return tags
    return ""


# Database Manager
class DatabaseManager:
    """Manages database operations for the agent steward."""

    def __init__(self, db_path: str = CONFIG["db_path"]):
        """Initialize the database manager."""
        self.db_path = db_path
        self._setup_database()

    def _setup_database(self) -> None:
        """Set up the database schema if it doesn't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Create memories table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS Memories (
                id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                text TEXT NOT NULL,
                createdBy TEXT NOT NULL,
                createdDate INTEGER NOT NULL,
                tags TEXT
            )
            """
            )

            # Create indexes for common queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON Memories(date)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_createdBy ON Memories(createdBy)"
            )
            cursor.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS MemoriesSearch USING fts5(id, text, tags)"
            )

            conn.commit()
            logger.info("Database initialized successfully")

    @contextmanager
    def get_connection(self):
        """Get a SQLite connection with appropriate timeout."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path, timeout=CONFIG["connection_timeout"], isolation_level=None
            )
            conn.execute("BEGIN")
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        else:
            if conn:
                conn.commit()
        finally:
            if conn:
                conn.close()

    def add_memory(
        self,
        text: str,
        tags: Union[str, List[str]] = "",
        created_by: str = "agent",
        date: Optional[str] = None,
    ) -> str:
        """Add a new memory to the database."""
        memory_id = nanoid_generate(size=10)
        normalized_date = get_formatted_date(date)
        normalized_tags = normalize_tags(tags)
        created_date = int(time.time() * 1000)

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Insert into main table
                cursor.execute(
                    "INSERT INTO Memories (id, date, text, createdBy, createdDate, tags) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        memory_id,
                        normalized_date,
                        text,
                        created_by,
                        created_date,
                        normalized_tags,
                    ),
                )

                # Insert into search table
                cursor.execute(
                    "INSERT INTO MemoriesSearch (id, text, tags) VALUES (?, ?, ?)",
                    (memory_id, text, normalized_tags),
                )

                logger.debug(f"Memory added: {memory_id} - {text[:50]}...")
                return memory_id
        except sqlite3.Error as e:
            logger.error(f"Error adding memory: {e}")
            raise RuntimeError(f"Failed to add memory: {str(e)}")

    def search_memories(
        self,
        date: Optional[str] = None,
        search_text: Optional[str] = None,
        tags: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for memories based on filters."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Build query based on provided filters
                query = "SELECT m.* FROM Memories m"
                params = []
                where_clauses = []

                # Date filter
                if date:
                    normalized_date = get_formatted_date(date)
                    where_clauses.append("m.date = ?")
                    params.append(normalized_date)

                # Text search
                if search_text:
                    # Use FTS5 for text search if available
                    query = """
                    SELECT m.* FROM Memories m
                    JOIN MemoriesSearch ms ON m.id = ms.id
                    """
                    where_clauses.append("ms.text MATCH ?")
                    params.append(search_text + "*")

                # Tags filter
                if tags:
                    normalized_tags = normalize_tags(tags)
                    tag_list = [tag.strip() for tag in normalized_tags.split(",")]

                    tag_conditions = []
                    for tag in tag_list:
                        tag_conditions.append("m.tags LIKE ?")
                        params.append(f"%{tag}%")

                    if tag_conditions:
                        where_clauses.append(f'({" OR ".join(tag_conditions)})')

                # Combine where clauses
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)

                # Order by date, most recent first
                query += " ORDER BY m.date DESC"

                logger.debug(f"Executing query: {query} with params: {params}")
                cursor.execute(query, params)
                results = cursor.fetchall()

                # Format results
                memories = []
                for row in results:
                    memories.append(
                        {
                            "id": row[0],
                            "date": row[1],
                            "text": row[2],
                            "createdBy": row[3],
                            "createdDate": row[4],
                            "tags": row[5],
                        }
                    )

                logger.debug(f"Found {len(memories)} memories")
                return memories
        except sqlite3.Error as e:
            logger.error(f"Error searching memories: {e}")
            raise RuntimeError(f"Failed to search memories: {str(e)}")

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from the database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM Memories WHERE id = ?", (memory_id,))
                cursor.execute("DELETE FROM MemoriesSearch WHERE id = ?", (memory_id,))

                if cursor.rowcount > 0:
                    logger.debug(f"Memory deleted: {memory_id}")
                    return True
                else:
                    logger.warning(f"Memory not found for deletion: {memory_id}")
                    return False
        except sqlite3.Error as e:
            logger.error(f"Error deleting memory: {e}")
            raise RuntimeError(f"Failed to delete memory: {str(e)}")

    def shutdown(self) -> None:
        """Clean shutdown of the database."""
        logger.info("Database shutdown complete")


# Initialize database manager
db_manager = DatabaseManager()


# Memory Tools
@tool
def make_note(
    text: str,
    tags: Union[str, List[str]] = "",
    created_by: str = "agent",
    date: Optional[str] = None,
) -> str:
    """Store information, events, or reminders in memory.

    Args:
        text: Content of the note (e.g., "Doctor appointment at 3pm")
        tags: Categories for organizing notes (e.g., "appointment,health")
        created_by: Source of the note (defaults to "agent")
        date: Date of the note in any common format (defaults to today)
    """
    try:
        if not text or not isinstance(text, str):
            return "Error: Note content is required and must be text"

        memory_id = db_manager.add_memory(
            text=text, tags=tags, created_by=created_by, date=date
        )

        return f"Note saved successfully with ID: {memory_id}"
    except Exception as e:
        logger.error(f"Error in make_note: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error saving note: {str(e)}"


@tool
def consult_notes(
    date: Optional[str] = None,
    search_text: Optional[str] = None,
    tags: Optional[Union[str, List[str]]] = None,
) -> str:
    """Search through stored notes by date, content, or tags.

    Args:
        date: Date to filter notes by (e.g., "2023-05-15", "today")
        search_text: Text to search for in note content
        tags: Categories to filter notes by (e.g., "meeting,work")
    """
    try:
        if not date and not search_text and not tags:
            return "Error: Please provide at least one search parameter (date, search_text, or tags)"

        memories = db_manager.search_memories(
            date=date, search_text=search_text, tags=tags
        )

        if not memories:
            return "No notes found matching your criteria."

        return json.dumps(memories, indent=2)
    except Exception as e:
        logger.error(f"Error in consult_notes: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error retrieving notes: {str(e)}"


# Improved Steward Setup
def setup_steward():
    """Set up the steward with Claude LLM and tools."""
    # Initialize Claude LLM
    logger.info(f"Initializing Claude with model: {CONFIG['model']}")
    llm = ChatAnthropic(
        model=CONFIG["model"],
        temperature=CONFIG["temperature"],
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    # Get current date information
    today = datetime.now()
    tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    # Define system message content
    system_prompt = f"""You are a personal assistant with a memory system. You have tools to store and retrieve information.

Today's date is {today.strftime("%Y-%m-%d")}. Tomorrow is {tomorrow}. Yesterday was {yesterday}.
Always convert relative dates (today, tomorrow) to specific dates (YYYY-MM-DD) when using tools.

Answer the following questions as best you can. You have access to the following tools:

{{tools}}

"""

    # Set up tools
    tools = [make_note, consult_notes]

    llm = llm.bind_tools(tools)
    # Create the agent using React framework with structured prompt
    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)

    logger.info("Steward setup complete")
    return agent


# Enhanced run function to execute a single query
def run_query(agent, query):
    """Run a single query through the agent and return the response."""
    try:
        logger.info(f"Processing query: {query}")

        # Format the query as a proper HumanMessage
        inputs = {"messages": [("user", query)]}
        # response = agent.invoke(inputs)
        for s in agent.stream(inputs, stream_mode="values"):
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

        response = ""
        logger.info("Query processed successfully")
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}"


# Main execution
def main():
    """Main execution function."""
    try:
        # Check for API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY environment variable is not set")
            print("Please set it with: export ANTHROPIC_API_KEY=your_api_key")
            return

        # Set up the assistant
        agent = setup_steward()
        print("Agent Steward initialized and ready")

        # Interactive loop
        print("\nEnter your queries below (type 'exit' to quit):")

        # Track conversation for context (just for display purposes)
        conversation = []

        response = run_query(
            agent, "Remind me about my doctor's appointment tomorrow at 2pm"
        )
        # while True:
        #     user_input = input("\nYou: ")
        #     if user_input.lower() in ["exit", "quit", "q"]:
        #         break
        #
        #     # Add to display conversation
        #     conversation.append(HumanMessage(content=user_input))
        #
        #     # Process with the agent
        #     response = run_query(agent, user_input)
        #
        #     # Add to display conversation
        #     conversation.append(AIMessage(content=response))
        #
        #     print(f"\nAssistant: {response}")

    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"An error occurred: {str(e)}")
    finally:
        # Clean shutdown
        db_manager.shutdown()
        print("\nAgent Steward shutdown complete")


if __name__ == "__main__":
    main()
