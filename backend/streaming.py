"""
Server-Sent Events (SSE) Streaming
Real-time event streaming for agent monitoring
"""

import asyncio
import json
from typing import AsyncGenerator, Dict, Any, Optional
from fastapi import Request
from fastapi.responses import StreamingResponse
from .agents.base_agent import AgentEvent, AgentEventBus

class EventStream:
    """SSE event stream manager"""
    
    def __init__(self, event_bus: AgentEventBus, job_id: str):
        self.event_bus = event_bus
        self.job_id = job_id
        self.queue: asyncio.Queue = asyncio.Queue()
        self._listener_task = None
        
    async def _event_listener(self, event: AgentEvent):
        """Listen to events from event bus"""
        await self.queue.put(event)
        
    async def start(self):
        """Start listening to events"""
        self.event_bus.subscribe(self._event_listener)
        
    async def stop(self):
        """Stop listening to events"""
        self.event_bus.unsubscribe(self._event_listener)
        
    async def stream(self) -> AsyncGenerator[str, None]:
        """Stream events as SSE"""
        try:
            await self.start()
            
            # Send initial connection message
            yield self._format_sse({
                "type": "connected",
                "job_id": self.job_id,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            while True:
                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(self.queue.get(), timeout=30.0)
                    
                    # Format and send event
                    yield self._format_sse(event.to_dict())
                    
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield self._format_sse({"type": "keepalive"})
                    
        except asyncio.CancelledError:
            # Client disconnected
            pass
        finally:
            await self.stop()
            
    def _format_sse(self, data: Dict[str, Any]) -> str:
        """Format data as SSE message"""
        return f"data: {json.dumps(data)}\n\n"

class StreamManager:
    """Manage multiple event streams"""
    
    def __init__(self):
        self.streams: Dict[str, EventStream] = {}
        
    def create_stream(self, job_id: str, event_bus: AgentEventBus) -> EventStream:
        """Create a new event stream"""
        stream = EventStream(event_bus, job_id)
        self.streams[job_id] = stream
        return stream
        
    def get_stream(self, job_id: str) -> Optional[EventStream]:
        """Get existing stream"""
        return self.streams.get(job_id)
        
    def remove_stream(self, job_id: str):
        """Remove stream"""
        if job_id in self.streams:
            del self.streams[job_id]

# Global stream manager
_stream_manager = None

def get_stream_manager() -> StreamManager:
    """Get or create global stream manager"""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager

async def create_sse_response(job_id: str, event_bus: AgentEventBus) -> StreamingResponse:
    """Create SSE streaming response"""
    manager = get_stream_manager()
    stream = manager.create_stream(job_id, event_bus)
    
    return StreamingResponse(
        stream.stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
