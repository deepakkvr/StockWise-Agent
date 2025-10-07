# agents/base_agent.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import asyncio
from loguru import logger

class AgentType(Enum):
    NEWS_ANALYZER = "news_analyzer"
    SENTIMENT_ANALYZER = "sentiment_analyzer" 
    SYNTHESIZER = "synthesizer"
    MARKET_ANALYZER = "market_analyzer"

class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentTask:
    """Represents a task for an agent to process"""
    task_id: str
    agent_type: AgentType
    input_data: Dict[str, Any]
    priority: int = 1
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    status: AgentStatus = AgentStatus.IDLE
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class AgentResult:
    """Standardized result format for all agents"""
    agent_type: AgentType
    task_id: str
    success: bool
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float
    timestamp: datetime
    confidence_score: Optional[float] = None
    sources: List[str] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []

class BaseAgent(ABC):
    """Base class for all StockWise agents"""
    
    def __init__(
        self,
        agent_type: AgentType,
        name: str,
        llm_provider: str = "mistral",
        max_concurrent_tasks: int = 3
    ):
        self.agent_type = agent_type
        self.name = name
        self.llm_provider = llm_provider
        self.max_concurrent_tasks = max_concurrent_tasks
        self.status = AgentStatus.IDLE
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []
        self.performance_metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        
        logger.info(f"Initialized {self.name} agent with {llm_provider} LLM")
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> AgentResult:
        """Process a single task - must be implemented by each agent"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for the agent"""
        pass
    
    async def submit_task(
        self,
        input_data: Dict[str, Any],
        task_id: Optional[str] = None,
        priority: int = 1
    ) -> str:
        """Submit a new task to the agent"""
        
        # Validate input
        if not self.validate_input(input_data):
            raise ValueError(f"Invalid input data for {self.name} agent")
        
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"{self.agent_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create task
        task = AgentTask(
            task_id=task_id,
            agent_type=self.agent_type,
            input_data=input_data,
            priority=priority
        )
        
        # Check concurrent task limit
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            logger.warning(f"{self.name} agent at max capacity, queuing task {task_id}")
        
        self.active_tasks[task_id] = task
        logger.info(f"Task {task_id} submitted to {self.name} agent")
        
        return task_id
    
    async def execute_task(self, task_id: str) -> AgentResult:
        """Execute a specific task"""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found in active tasks")
        
        task = self.active_tasks[task_id]
        task.status = AgentStatus.PROCESSING
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting task {task_id} in {self.name} agent")
            
            # Process the task
            result = await self.process_task(task)
            
            # Update task status
            task.status = AgentStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result.data
            
            # Update performance metrics
            processing_time = (task.completed_at - start_time).total_seconds()
            self._update_performance_metrics(processing_time, success=True)
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            del self.active_tasks[task_id]
            
            logger.info(f"Task {task_id} completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            # Handle task failure
            task.status = AgentStatus.ERROR
            task.error_message = str(e)
            task.completed_at = datetime.now()
            
            processing_time = (task.completed_at - start_time).total_seconds()
            self._update_performance_metrics(processing_time, success=False)
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            del self.active_tasks[task_id]
            
            logger.error(f"Task {task_id} failed: {e}")
            
            # Return error result
            return AgentResult(
                agent_type=self.agent_type,
                task_id=task_id,
                success=False,
                data={'error': str(e)},
                metadata={'error_type': type(e).__name__},
                processing_time=processing_time,
                timestamp=datetime.now()
            )
    
    async def process_batch(
        self,
        batch_data: List[Dict[str, Any]],
        batch_id: Optional[str] = None
    ) -> List[AgentResult]:
        """Process multiple tasks concurrently"""
        
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Processing batch {batch_id} with {len(batch_data)} tasks")
        
        # Submit all tasks
        task_ids = []
        for i, data in enumerate(batch_data):
            task_id = f"{batch_id}_task_{i}"
            await self.submit_task(data, task_id)
            task_ids.append(task_id)
        
        # Execute tasks concurrently
        tasks = [self.execute_task(task_id) for task_id in task_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        final_results = []
        for task_id, result in zip(task_ids, results):
            if isinstance(result, Exception):
                logger.error(f"Batch task {task_id} failed: {result}")
                final_results.append(AgentResult(
                    agent_type=self.agent_type,
                    task_id=task_id,
                    success=False,
                    data={'error': str(result)},
                    metadata={'batch_id': batch_id},
                    processing_time=0.0,
                    timestamp=datetime.now()
                ))
            else:
                final_results.append(result)
        
        successful_tasks = sum(1 for r in final_results if r.success)
        logger.info(f"Batch {batch_id} completed: {successful_tasks}/{len(batch_data)} successful")
        
        return final_results
    
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """Update internal performance metrics"""
        if success:
            self.performance_metrics['tasks_completed'] += 1
        else:
            self.performance_metrics['tasks_failed'] += 1
        
        self.performance_metrics['total_processing_time'] += processing_time
        
        total_tasks = (self.performance_metrics['tasks_completed'] + 
                      self.performance_metrics['tasks_failed'])
        
        if total_tasks > 0:
            self.performance_metrics['average_processing_time'] = (
                self.performance_metrics['total_processing_time'] / total_tasks
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            'agent_name': self.name,
            'agent_type': self.agent_type.value,
            'status': self.status.value,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'performance_metrics': self.performance_metrics.copy(),
            'llm_provider': self.llm_provider
        }
    
    def clear_completed_tasks(self, keep_last: int = 100):
        """Clear old completed tasks to free memory"""
        if len(self.completed_tasks) > keep_last:
            removed = len(self.completed_tasks) - keep_last
            self.completed_tasks = self.completed_tasks[-keep_last:]
            logger.info(f"Cleared {removed} old completed tasks from {self.name} agent")
    
    def get_task_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent task history"""
        recent_tasks = self.completed_tasks[-limit:] if self.completed_tasks else []
        
        history = []
        for task in recent_tasks:
            history.append({
                'task_id': task.task_id,
                'status': task.status.value,
                'created_at': task.created_at.isoformat(),
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'processing_time': (
                    (task.completed_at - task.created_at).total_seconds() 
                    if task.completed_at else None
                ),
                'success': task.status == AgentStatus.COMPLETED,
                'error_message': task.error_message
            })
        return history

