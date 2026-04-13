from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class UserContent:
    """
    A dataclass to hold user content for AI model calls.
    """
    temperature: float = 0.1
    system_prompt: Optional[str] = None
    text: Optional[str] = None
    images: List[str] = field(default_factory=list) # Base64 strings
    
    def has_images(self) -> bool:
        return len(self.images) > 0