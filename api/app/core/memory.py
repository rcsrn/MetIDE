from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from typing import List

from app.core.supabase_client import supabase

class SupabaseChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores messages in a Supabase table."""

    def __init__(self, session_id: str, table_name: str = "historial_chat") -> None:
        """Initialize with session ID and Supabase table name."""
        self.session_id = session_id
        self.table_name = table_name
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages from Supabase"""
        response = supabase.table("historial_chat").select("historial").eq("session_id", self.session_id).execute()
        if not response.data:
            return []

        items = response.data[0].get("historial", [])
        return messages_from_dict(items)

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        supabase.table(self.table_name).insert({
            "session_id": self.session_id,
            "message": message_to_dict(message)
        }).execute()

    def clear(self) -> None:
        """Clear all messages in the history for the current session."""
        supabase.table(self.table_name).delete().eq("session_id", self.session_id).execute()

    def get_messages(self) -> List[BaseMessage]:
        """Get all messages in the history for the current session."""
        response = supabase.table(self.table_name).select("message").eq("session_id", self.session_id).order("id", ascending=True).execute()
        records = response.data
        return messages_from_dict([record["message"] for record in records]) if records else []