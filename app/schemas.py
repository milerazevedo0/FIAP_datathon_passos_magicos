from pydantic import BaseModel

class StudentInput(BaseModel):
    IAA: float | None = None
    IEG: float | None = None
    IPS: float | None = None
    IPP: float | None = None
    IDA: float | None = None
    IPV: float | None = None
    IAN: float | None = None
    PORTUGUES: float | None = None
    MATEMATICA: float | None = None
    INGLES: float | None = None
