from typing import Annotated, List, TypedDict
import operator

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    next_action: str
    user_id: str