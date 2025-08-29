from typing import List, Optional
from pydantic import BaseModel, Field


class Company(BaseModel):
    name: str
    website: str
    why_fit: str


class CompaniesPayload(BaseModel):
    companies: List[Company] = Field(default_factory=list)


class Contact(BaseModel):
    name: str
    role: str
    email: Optional[str] = None
    inferred: Optional[bool] = False
    source: Optional[str] = None


class ContactsPerCompany(BaseModel):
    company: str
    contacts: List[Contact] = Field(default_factory=list)


class ContactsPayload(BaseModel):
    companies: List[ContactsPerCompany] = Field(default_factory=list)


class ResearchPerCompany(BaseModel):
    company: str
    insights: List[str] = Field(default_factory=list)


class ResearchPayload(BaseModel):
    companies: List[ResearchPerCompany] = Field(default_factory=list)


class EmailItem(BaseModel):
    company: str
    contact: str
    subject: str
    body: str


class EmailsPayload(BaseModel):
    emails: List[EmailItem] = Field(default_factory=list)
