from __future__ import annotations
import logging,os,re,time,threading
from collections import OrderedDict
from dataclasses import dataclass,field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

FAILURE="I couldn't get a reliable answer from the web right now."
@dataclass
class WebAnswer:
    answer:str; success:bool; sources:list[dict]=field(default_factory=list); model:str=""; web_request_ms:float=0; answer_processing_ms:float=0; error:str|None=None;cache_hit:bool=False;input_tokens:int=0;output_tokens:int=0

class WebAnswerService:
    def __init__(self,client=None,model=None,search_context=None,timeout=None):
        self.client=client;self.model=model or os.getenv("JARVIS_WEB_ANSWER_MODEL","gpt-5.4-mini");self.search_context=search_context or os.getenv("JARVIS_WEB_SEARCH_CONTEXT","low");self.timeout=float(timeout or os.getenv("JARVIS_WEB_ANSWER_TIMEOUT","12"));self.log=logging.getLogger("jarvis.web_answer")
        self.cache_max=max(0,int(os.getenv("JARVIS_WEB_CACHE_MAX","128")));self.cache_ttl=max(0,float(os.getenv("JARVIS_WEB_CACHE_STABLE_TTL","86400")));self._cache=OrderedDict();self._cache_lock=threading.Lock()
    def answer(self,question,cancellation_token=None):
        started=time.perf_counter()
        try:
            if cancellation_token is not None:cancellation_token.raise_if_cancelled()
            cached=self._cache_get(question)
            if cached is not None:
                if cancellation_token is not None:cancellation_token.raise_if_cancelled()
                return WebAnswer(cached.answer,True,list(cached.sources),cached.model,0,0,cache_hit=True)
            if self.client is None:
                self.client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response=self.client.responses.create(model=self.model,tools=[{"type":"web_search","search_context_size":self.search_context}],include=["web_search_call.action.sources"],input=[{"role":"system","content":"Answer in English using current web information. Reply in one to three concise natural sentences suitable for speech. Do not include Markdown, raw URLs, or a spoken source list."},{"role":"user","content":question}],max_output_tokens=180,store=False,timeout=self.timeout)
            if cancellation_token is not None:cancellation_token.raise_if_cancelled()
            web_ms=(time.perf_counter()-started)*1000; processing=time.perf_counter(); text=self._spoken_text(getattr(response,"output_text","") or ""); sources=self._sources(response); processing_ms=(time.perf_counter()-processing)*1000
            if not text:return WebAnswer(FAILURE,False,sources,self.model,web_ms,processing_ms,"empty_response")
            usage=getattr(response,"usage",None);result=WebAnswer(text,True,sources,self.model,web_ms,processing_ms,input_tokens=getattr(usage,"input_tokens",0) or 0,output_tokens=getattr(usage,"output_tokens",0) or 0);self._cache_put(question,result)
            self.log.info("Web answer performance: web_request_ms=%.1f answer_processing_ms=%.1f model=%s sources=%d",web_ms,processing_ms,self.model,len(sources));return result
        except Exception as exc:
            if cancellation_token is not None and cancellation_token.cancelled:
                raise
            elapsed=(time.perf_counter()-started)*1000;self.log.exception("Web answer request failed")
            return WebAnswer(FAILURE,False,[],self.model,elapsed,0,f"{type(exc).__name__}: {exc}")
    @staticmethod
    def _cacheable(question):
        text=" ".join(question.lower().split())
        volatile=r"\b(?:today|tonight|yesterday|tomorrow|now|current|currently|latest|recent|news|weather|forecast|price|stock|score|schedule|president|prime minister|ceo|version|release)\b"
        return bool(text) and not re.search(volatile,text)
    def _cache_get(self,question):
        if not self.cache_max or not self.cache_ttl or not self._cacheable(question):return None
        key=" ".join(question.lower().split());now=time.monotonic()
        with self._cache_lock:
            item=self._cache.get(key)
            if item is None:return None
            expires,result=item
            if expires<=now:self._cache.pop(key,None);return None
            self._cache.move_to_end(key);return result
    def _cache_put(self,question,result):
        if not self.cache_max or not self.cache_ttl or not self._cacheable(question):return
        key=" ".join(question.lower().split())
        with self._cache_lock:
            self._cache[key]=(time.monotonic()+self.cache_ttl,result);self._cache.move_to_end(key)
            while len(self._cache)>self.cache_max:self._cache.popitem(last=False)
    @staticmethod
    def _spoken_text(text):
        text=re.sub(r"【[^】]+】", "", text);text=re.sub(r"\(?\[[^\]]+\]\(https?://[^)]+\)\)?", "", text);text=re.sub(r"https?://\S+", "", text);text=re.sub(r"\[(?:\d+|source[^\]]*)\]", "", text,flags=re.I);text=re.sub(r"[*_`#]", "", text);return re.sub(r"\s+", " ", text).strip()
    @staticmethod
    def _sources(response):
        if hasattr(response,"model_dump"):
            try:data=response.model_dump(warnings=False)
            except TypeError:data=response.model_dump()
        else:data=response if isinstance(response,dict) else {}
        found={}
        def walk(value):
            if isinstance(value,dict):
                url=value.get("url")
                if isinstance(url,str) and url.startswith("http"):found[url]={"url":url,"title":value.get("title")}
                for child in value.values():walk(child)
            elif isinstance(value,list):
                for child in value:walk(child)
        walk(data);return list(found.values())

_SERVICE=None
def get_web_answer_service():
    global _SERVICE
    if _SERVICE is None:_SERVICE=WebAnswerService()
    return _SERVICE
