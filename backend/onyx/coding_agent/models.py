from pydantic import BaseModel


class CodingAgentArguments(BaseModel):
    query: str
    github_repo: str
