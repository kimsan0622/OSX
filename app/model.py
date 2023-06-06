import os
import datetime
import pytz
from pymongo import MongoClient
from bson.codec_options import CodecOptions


mongo_client = MongoClient(os.getenv('MONGODB_URI', "mongodb://root:1234@10.1.92.1:27018"))

NODE_TIMEZONE = pytz.timezone('Asia/Seoul')

class DBManager(object):
    def __init__(self, job_name) -> None:
        self.db = mongo_client[job_name]


# state: 'new' --> 'downloaded, -ing' --> 'asr_done, -ing' --> 'split_done, ing'
class YoutubeLinks():

    __collection_name__="YoutubeLinks"

    def __init__(self, db, tzinfo=None):
        if tzinfo is not None:
            self.collection = db[YoutubeLinks.__collection_name__].with_options(
                codec_options=CodecOptions(
                    tz_aware=True,
                    tzinfo=tzinfo
                )
            )
        else:
            self.collection = db[YoutubeLinks.__collection_name__]
    
    def check_docs(self, url):
        return self.collection.count_documents({'url': url})
    
    def add_document(self, url, state="new"):
        # if the topic doesn't exists, insert the topic
        if not self.check_docs(url):
            self.collection.insert_one({
                "url": url,
                "video_path": "",
                "audio_path": "",
                "asr_path": "",
                "state": state,
                "error_msg": "",
                "duration": "",
                "last_modified": datetime.datetime.utcnow()
            })
    
    def add_document_with_video_path(self, url, video_path, duration="", error_msg="", state="downloaded"):
        # if the topic doesn't exists, insert the topic
        if not self.check_docs(url):
            self.collection.insert_one({
                "url": url,
                "video_path": video_path,
                "audio_path": "",
                "asr_path": "",
                "state": state,
                "error_msg": "",
                "duration": duration,
                "last_modified": datetime.datetime.utcnow()
            })
        else:
            self.collection.update_one(
                {'url': url}, 
                {'$set': {
                    "video_path": video_path,
                    "state": state,
                    "error_msg": error_msg,
                    "duration": duration,
                    "last_modified": datetime.datetime.utcnow()
                    }
                },
            )
            
            
    def set_state(self, url, state="downloading"):
        if self.check_docs(url):
            self.collection.update_one(
                {'url': url}, 
                {'$set': {
                    "state": state,
                    "last_modified": datetime.datetime.utcnow()
                    }
                },
            )
            
            
    def set_state_and_video_path(self, url, video_path, duration="", state="downloaded"):
        if self.check_docs(url):
            self.collection.update_one(
                {'url': url}, 
                {'$set': {
                    "video_path": video_path,
                    "state": state,
                    "duration": duration,
                    "last_modified": datetime.datetime.utcnow()
                    }
                },
            )
    
    def set_state_and_error_msg(self, url, error_msg, state="error"):
        if self.check_docs(url):
            self.collection.update_one(
                {'url': url}, 
                {'$set': {
                    "error_msg": error_msg,
                    "state": state,
                    "last_modified": datetime.datetime.utcnow()
                    }
                },
            )
    
    def set_state_and_audio_path(self, url, audio_path, asr_path, state="asr_done"):
        if self.check_docs(url):
            self.collection.update_one(
                {'url': url}, 
                {'$set': {
                    "audio_path": audio_path,
                    "asr_path": asr_path,
                    "state": state,
                    "last_modified": datetime.datetime.utcnow()
                    }
                },
            )
    
    def check_docs(self, url):
        return self.collection.count_documents({'url': url})
    
    def get_all_documents(self, dtToString=False):
        cursor = self.collection.find({})
        return [{
                    "url": doc["url"],
                    "video_path": doc["video_path"],
                    "audio_path": doc["audio_path"],
                    "asr_path": doc["asr_path"],
                    "state": doc["state"],
                    "error_msg": doc["error_msg"],
                    "duration": doc["duration"],
                    "last_modified": str(doc["last_modified"]) if dtToString else doc["last_modified"],
                } for doc in cursor]
    
    def get_all_documents_by_state(self, state="new", dtToString=False):
        cursor = self.collection.find({"state": state})
        return [{
                    "url": doc["url"],
                    "video_path": doc["video_path"],
                    "audio_path": doc["audio_path"],
                    "asr_path": doc["asr_path"],
                    "state": doc["state"],
                    "error_msg": doc["error_msg"],
                    "duration": doc["duration"],
                    "last_modified": str(doc["last_modified"]) if dtToString else doc["last_modified"],
                } for doc in cursor]
    
    def get_single_document_by_state(self, state="new", dtToString=False):
        doc = self.collection.find_one({"state": state})
        if doc is not None:
            return {
                        "url": doc["url"],
                        "video_path": doc["video_path"],
                        "audio_path": doc["audio_path"],
                        "asr_path": doc["asr_path"],
                        "state": doc["state"],
                        "error_msg": doc["error_msg"],
                        "duration": doc["duration"],
                        "last_modified": str(doc["last_modified"]) if dtToString else doc["last_modified"],
                    }
        return None


class YoutubeClips():

    __collection_name__="YoutubeClips"

    def __init__(self, db, tzinfo=None):
        if tzinfo is not None:
            self.collection = db[YoutubeClips.__collection_name__].with_options(
                codec_options=CodecOptions(
                    tz_aware=True,
                    tzinfo=tzinfo
                )
            )
        else:
            self.collection = db[YoutubeClips.__collection_name__]
    
    def check_docs(self, clip_id):
        return self.collection.count_documents({'clip_id': clip_id})
    
    def add_document(
            self, 
            clip_id, 
            clip_video_path, 
            clip_info_path, 
            clip_audio_path, 
            text, 
            origin_url, 
            start_time,
            end_time,
            state="new"
        ):
        if not self.check_docs(clip_id):
            self.collection.insert_one({
                "clip_id": clip_id,
                "clip_video_path": clip_video_path,
                "clip_info_path": clip_info_path,
                "clip_audio_path": clip_audio_path,
                "text": text,
                "url": origin_url,
                "start_time": start_time,
                "end_time": end_time,
                "state": state,
                "last_modified": datetime.datetime.utcnow()
            })
        else:
            self.collection.update_one(
                {'clip_id': clip_id}, 
                {'$set': {
                    "clip_video_path": clip_video_path,
                    "clip_info_path": clip_info_path,
                    "clip_audio_path": clip_audio_path,
                    "text": text,
                    "url": origin_url,
                    "start_time": start_time,
                    "end_time": end_time,
                    "state": state,
                    "last_modified": datetime.datetime.utcnow()
                    }
                },
            )
    
    def get_all_documents(self, dtToString=False):
        cursor = self.collection.find({})
        return [{
                    "clip_id": doc["clip_id"],
                    "clip_video_path": doc["clip_video_path"],
                    "clip_info_path": doc["clip_info_path"],
                    "clip_audio_path": doc["clip_audio_path"],
                    "text": doc["text"],
                    "url": doc["url"],
                    "start_time": doc["start_time"],
                    "end_time": doc["end_time"],
                    "state": doc["state"],
                    "last_modified": str(doc["last_modified"]) if dtToString else doc["last_modified"],
                } for doc in cursor]
    
    def get_all_documents_by_state(self, state="new", dtToString=False):
        cursor = self.collection.find({"state": state})
        return [{
                    "clip_id": doc["clip_id"],
                    "clip_video_path": doc["clip_video_path"],
                    "clip_info_path": doc["clip_info_path"],
                    "clip_audio_path": doc["clip_audio_path"],
                    "text": doc["text"],
                    "url": doc["url"],
                    "start_time": doc["start_time"],
                    "end_time": doc["end_time"],
                    "state": doc["state"],
                    "last_modified": str(doc["last_modified"]) if dtToString else doc["last_modified"],
                } for doc in cursor]
    
    def get_document_by_clip_id(self, clip_id, dtToString=False):
        doc = self.collection.find_one({"clip_id": clip_id})
        return {
                    "clip_id": doc["clip_id"],
                    "clip_video_path": doc["clip_video_path"],
                    "clip_info_path": doc["clip_info_path"],
                    "clip_audio_path": doc["clip_audio_path"],
                    "text": doc["text"],
                    "url": doc["url"],
                    "start_time": doc["start_time"],
                    "end_time": doc["end_time"],
                    "state": doc["state"],
                    "last_modified": str(doc["last_modified"]) if dtToString else doc["last_modified"],
                }
    
    def get_document_by_state(self, state="new", dtToString=False):
        doc = self.collection.find_one({"state": state})
        return {
                    "clip_id": doc["clip_id"],
                    "clip_video_path": doc["clip_video_path"],
                    "clip_info_path": doc["clip_info_path"],
                    "clip_audio_path": doc["clip_audio_path"],
                    "text": doc["text"],
                    "url": doc["url"],
                    "start_time": doc["start_time"],
                    "end_time": doc["end_time"],
                    "state": doc["state"],
                    "last_modified": str(doc["last_modified"]) if dtToString else doc["last_modified"],
                }
    
    def set_state(self, clip_id, state="new"):
        if self.check_docs(clip_id):
            self.collection.update_one(
                {'clip_id': clip_id}, 
                {'$set': {
                    "state": state,
                    "last_modified": datetime.datetime.utcnow()
                    }
                },
            )
    
    def get_random_clip(self, dtToString=False):
        cursor = self.collection.aggregate([{'$sample':{'size':1}}])
        return [{
            "clip_id": doc["clip_id"],
            "clip_video_path": doc["clip_video_path"],
            "clip_info_path": doc["clip_info_path"],
            "clip_audio_path": doc["clip_audio_path"],
            "text": doc["text"],
            "url": doc["url"],
            "start_time": doc["start_time"],
            "end_time": doc["end_time"],
            "state": doc["state"],
            "last_modified": str(doc["last_modified"]) if dtToString else doc["last_modified"],
        } for doc in cursor][0]
    
    def get_random_clip_by_state_and_undefined_field(self, state, undefined_field, dtToString=False):
        cursor = self.collection.aggregate([
            {'$match': {'$and': [{'state': state},{undefined_field: None}]}},
            {'$sample':{'size':1}}
        ])
        
        res = [{
            "clip_id": doc["clip_id"],
            "clip_video_path": doc["clip_video_path"],
            "clip_info_path": doc["clip_info_path"],
            "clip_audio_path": doc["clip_audio_path"],
            "text": doc["text"],
            "url": doc["url"],
            "start_time": doc["start_time"],
            "end_time": doc["end_time"],
            "state": doc["state"],
            "last_modified": str(doc["last_modified"]) if dtToString else doc["last_modified"],
        } for doc in cursor]
        if len(res) > 0:
            return res[0]
        else:
            return None
    
    def get_random_clip_by_undefined_field(self, undefined_field, dtToString=False):
        cursor = self.collection.aggregate([
            {'$match': {undefined_field: None}},
            {'$sample':{'size':1}}
        ])
        
        res = [{
            "clip_id": doc["clip_id"],
            "clip_video_path": doc["clip_video_path"],
            "clip_info_path": doc["clip_info_path"],
            "clip_audio_path": doc["clip_audio_path"],
            "text": doc["text"],
            "url": doc["url"],
            "start_time": doc["start_time"],
            "end_time": doc["end_time"],
            "state": doc["state"],
            "last_modified": str(doc["last_modified"]) if dtToString else doc["last_modified"],
        } for doc in cursor]
        if len(res) > 0:
            return res[0]
        else:
            return None
    
    def find_all_docs_by_state_and_undefined_field(self, state, undefined_field, dtToString=False):
        cursor = self.collection.find({'$and': [
            {'state': state},
            {undefined_field: None}]})
        return [{
                    "clip_id": doc["clip_id"],
                    "clip_video_path": doc["clip_video_path"],
                    "clip_info_path": doc["clip_info_path"],
                    "clip_audio_path": doc["clip_audio_path"],
                    "text": doc["text"],
                    "url": doc["url"],
                    "start_time": doc["start_time"],
                    "end_time": doc["end_time"],
                    "state": doc["state"],
                    "last_modified": str(doc["last_modified"]) if dtToString else doc["last_modified"],
                } for doc in cursor]
    
    def find_single_doc_by_state_and_undefined_field(self, state, undefined_field, dtToString=False):
        doc = self.collection.find_one({'$and': [
            {'state': state},
            {undefined_field: {'$exists': False}}]})
        return {
                    "clip_id": doc["clip_id"],
                    "clip_video_path": doc["clip_video_path"],
                    "clip_info_path": doc["clip_info_path"],
                    "clip_audio_path": doc["clip_audio_path"],
                    "text": doc["text"],
                    "url": doc["url"],
                    "start_time": doc["start_time"],
                    "end_time": doc["end_time"],
                    "state": doc["state"],
                    "last_modified": str(doc["last_modified"]) if dtToString else doc["last_modified"],
                }
    
    def update_undefined_field_by_clip_id(self, clip_id, field, value, **kwargs):
        self.collection.update_one(
                {'clip_id': clip_id}, 
                {'$set': {
                    field: value,
                    **kwargs,
                    "last_modified": datetime.datetime.utcnow()
                    }
                },
            )


class YoutubeDBManager(DBManager):
    def __init__(self, db_name="collect_youtube", tzinfo=NODE_TIMEZONE) -> None:
        self.db = mongo_client[db_name]
        
        self.youtube_links = YoutubeLinks(self.db, tzinfo)
        self.youtube_clips = YoutubeClips(self.db, tzinfo)
