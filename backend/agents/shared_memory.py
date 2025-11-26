"""
Shared Memory for Multi-Agent Communication
Allows agents to share data and coordinate
"""

import asyncio
from typing import Dict, Any, List, Optional
from collections import defaultdict

class SharedMemory:
    """Shared memory space for agents to communicate"""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._subscribers: Dict[str, List[callable]] = defaultdict(list)
        
    async def set(self, key: str, value: Any):
        """Set a value in shared memory"""
        async with self._locks[key]:
            self._data[key] = value
            
            # Notify subscribers
            for callback in self._subscribers[key]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(key, value)
                    else:
                        callback(key, value)
                except Exception as e:
                    print(f"Error in subscriber callback: {e}")
                    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from shared memory"""
        async with self._locks[key]:
            return self._data.get(key, default)
            
    async def append(self, key: str, value: Any):
        """Append to a list in shared memory"""
        async with self._locks[key]:
            if key not in self._data:
                self._data[key] = []
            if not isinstance(self._data[key], list):
                raise ValueError(f"Key {key} is not a list")
            self._data[key].append(value)
            
    async def extend(self, key: str, values: List[Any]):
        """Extend a list in shared memory"""
        async with self._locks[key]:
            if key not in self._data:
                self._data[key] = []
            if not isinstance(self._data[key], list):
                raise ValueError(f"Key {key} is not a list")
            self._data[key].extend(values)
            
    async def increment(self, key: str, amount: int = 1):
        """Increment a counter in shared memory"""
        async with self._locks[key]:
            if key not in self._data:
                self._data[key] = 0
            self._data[key] += amount
            return self._data[key]
            
    def subscribe(self, key: str, callback: callable):
        """Subscribe to changes on a key"""
        self._subscribers[key].append(callback)
        
    def unsubscribe(self, key: str, callback: callable):
        """Unsubscribe from changes on a key"""
        if callback in self._subscribers[key]:
            self._subscribers[key].remove(callback)
            
    async def clear(self):
        """Clear all data"""
        for key in list(self._data.keys()):
            async with self._locks[key]:
                del self._data[key]
                
    async def get_all(self) -> Dict[str, Any]:
        """Get all data (snapshot)"""
        return dict(self._data)
