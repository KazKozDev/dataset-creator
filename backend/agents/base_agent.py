"""
Base Agent Framework for Multi-Agent Generation
Provides event emission and coordination for advanced generation methods
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

class AgentRole(Enum):
    """Agent roles for different generation methods"""
    # Swarm roles
    SCOUT = "scout"
    GATHERER = "gatherer"
    MUTATOR = "mutator"
    SELECTOR = "selector"
    
    # Evolution roles
    MUTAGEN = "mutagen"
    CROSSOVER = "crossover"
    
    # Cosmic roles
    EXPLODER = "exploder"
    COOLER = "cooler"
    SYNTHESIZER = "synthesizer"
    
    # Quantum roles
    GAUGE = "gauge"
    FERMION = "fermion"
    YUKAWA = "yukawa"
    HIGGS = "higgs"
    POTENTIAL = "potential"
    
    # Multi-model adversarial roles
    GENERATOR = "generator"      # Creates initial content
    CRITIC = "critic"            # Finds problems/weaknesses
    REFINER = "refiner"          # Improves based on criticism
    DIVERSITY_CHECKER = "diversity_checker"  # Ensures uniqueness

class EventType(Enum):
    """Event types for agent communication"""
    START = "start"
    THINKING = "thinking"
    STREAMING = "streaming"
    OUTPUT = "output"
    COMPLETE = "complete"
    ERROR = "error"
    METADATA = "metadata"

@dataclass
class AgentEvent:
    """Event emitted by an agent"""
    agent_id: str
    agent_role: AgentRole
    event_type: EventType
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role.value,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data
        }

class AgentEventBus:
    """Event bus for agent communication and monitoring"""
    
    def __init__(self):
        self.listeners: List[Callable] = []
        self.event_history: List[AgentEvent] = []
        
    def subscribe(self, listener: Callable):
        """Subscribe to agent events"""
        self.listeners.append(listener)
        
    def unsubscribe(self, listener: Callable):
        """Unsubscribe from agent events"""
        if listener in self.listeners:
            self.listeners.remove(listener)
            
    async def emit(self, event: AgentEvent):
        """Emit an event to all listeners"""
        self.event_history.append(event)
        
        # Notify all listeners
        for listener in self.listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                else:
                    listener(event)
            except Exception as e:
                print(f"Error in event listener: {e}")
                
    def get_history(self, agent_id: Optional[str] = None) -> List[AgentEvent]:
        """Get event history, optionally filtered by agent_id"""
        if agent_id:
            return [e for e in self.event_history if e.agent_id == agent_id]
        return self.event_history
    
    def clear_history(self):
        """Clear event history"""
        self.event_history = []

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        event_bus: AgentEventBus,
        llm_provider: str = "ollama",
        llm_model: str = "llama3.2",
        temperature: float = 0.7
    ):
        self.agent_id = agent_id
        self.role = role
        self.event_bus = event_bus
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.temperature = temperature
        self.start_time = None
        self.end_time = None
        
    async def emit_event(self, event_type: EventType, data: Dict[str, Any]):
        """Emit an event"""
        event = AgentEvent(
            agent_id=self.agent_id,
            agent_role=self.role,
            event_type=event_type,
            timestamp=time.time(),
            data=data
        )
        await self.event_bus.emit(event)
        
    async def start(self):
        """Signal agent start"""
        self.start_time = time.time()
        await self.emit_event(EventType.START, {
            "message": f"{self.role.value} agent starting",
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model
        })
        
    async def think(self, thought: str):
        """Emit a thinking event"""
        await self.emit_event(EventType.THINKING, {
            "thought": thought
        })
        
    async def stream(self, partial_output: str):
        """Emit a streaming event with partial output"""
        await self.emit_event(EventType.STREAMING, {
            "partial_output": partial_output
        })
        
    async def output(self, result: Any, metadata: Optional[Dict] = None):
        """Emit an output event"""
        await self.emit_event(EventType.OUTPUT, {
            "result": result,
            "metadata": metadata or {}
        })
        
    async def complete(self, summary: Dict[str, Any]):
        """Signal agent completion"""
        self.end_time = time.time()
        duration = self.end_time - self.start_time if self.start_time else 0
        
        await self.emit_event(EventType.COMPLETE, {
            "message": f"{self.role.value} agent completed",
            "duration": duration,
            **summary
        })
        
    async def error(self, error_message: str, error_details: Optional[Dict] = None):
        """Emit an error event"""
        await self.emit_event(EventType.ERROR, {
            "error": error_message,
            "details": error_details or {}
        })
        
    async def metadata(self, metadata: Dict[str, Any]):
        """Emit metadata (tokens, cost, etc.)"""
        await self.emit_event(EventType.METADATA, metadata)
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using LLM
        Override this in subclasses for specific behavior
        """
        raise NotImplementedError("Subclasses must implement generate()")
        
    async def run(self, task: Dict[str, Any]) -> Any:
        """
        Main execution method
        Override this in subclasses
        """
        raise NotImplementedError("Subclasses must implement run()")

# Global event bus instance
_global_event_bus = None

# LLM call budget tracker (per-task)
class LLMBudget:
    """
    Tracks LLM call budget for a generation task.
    Budget = requested_count * multiplier (default 2x for retries/variations)
    """
    
    def __init__(self, requested_count: int, multiplier: float = 2.0, max_absolute: int = 100):
        # Budget is proportional to requested examples, with absolute cap
        self.requested_count = requested_count
        self.budget = min(int(requested_count * multiplier), max_absolute)
        self.call_count = 0
        self.max_absolute = max_absolute
        
    def can_call(self) -> bool:
        """Check if we have budget for another LLM call"""
        return self.call_count < self.budget
        
    def use(self) -> bool:
        """Use one call from budget. Returns True if successful."""
        if self.can_call():
            self.call_count += 1
            return True
        return False
        
    def remaining(self) -> int:
        """Get remaining budget"""
        return max(0, self.budget - self.call_count)
    
    def status(self) -> str:
        """Get status string"""
        return f"{self.call_count}/{self.budget} calls (for {self.requested_count} examples)"

# Task-scoped budget (set at start of each generation task)
_current_budget: Optional[LLMBudget] = None

def get_event_bus() -> AgentEventBus:
    """Get or create global event bus"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = AgentEventBus()
    return _global_event_bus

def set_llm_budget(requested_count: int, multiplier: float = 2.0, max_absolute: int = 100) -> LLMBudget:
    """Set LLM budget for current task based on requested example count"""
    global _current_budget
    _current_budget = LLMBudget(requested_count, multiplier, max_absolute)
    return _current_budget

def get_llm_budget() -> Optional[LLMBudget]:
    """Get current LLM budget (None if not set)"""
    return _current_budget

def clear_llm_budget():
    """Clear current budget (call after task completes)"""
    global _current_budget
    _current_budget = None

# Legacy compatibility
def get_llm_limiter(max_calls: int = 50) -> LLMBudget:
    """Legacy: Get or create LLM budget"""
    global _current_budget
    if _current_budget is None:
        _current_budget = LLMBudget(max_calls // 2, 2.0, max_calls)
    return _current_budget

def reset_llm_limiter():
    """Legacy: Reset budget"""
    clear_llm_budget()
