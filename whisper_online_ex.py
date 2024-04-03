from whisper_online import *

class FasterWhisperASREx(FasterWhisperASR):
    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr,
                 model=None, local_files_only=True, device="cuda", compute_type="float16"):
        self.logfile = logfile

        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan
        
        self.local_file_only = local_files_only
        self.device = device
        self.compute_type = compute_type
        self.model = model
        if model is None:
            self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel
        if model_dir is not None:
            print(f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used.",file=self.logfile)
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")


        # this worked fast and reliably on NVIDIA L40
        model = WhisperModel(model_size_or_path, 
                             device=self.device,
                            compute_type=self.compute_type, 
                            download_root=cache_dir,
                            local_files_only=self.local_file_only)

        # or run on GPU with INT8
        # tested: the transcripts were different, probably worse than with FP16, and it was slightly (appx 20%) slower
        #model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

        # or run on CPU with INT8
        # tested: works, but slow, appx 10-times than cuda FP16
#        model = WhisperModel(modelsize, device="cpu", compute_type="int8") #, download_root="faster-disk-cache-dir/")
        return model

class OnlineASRProcessorEx(OnlineASRProcessor):
    def duration(self):
        return len(self.audio_buffer)/self.SAMPLING_RATE
    
    def process_iter(self, stream_close=False):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, ""). 
        The non-emty text is confirmed (committed) partial transcript.
        """

        prompt, non_prompt = self.prompt()
        print("PROMPT:", prompt, file=self.logfile)
        print("CONTEXT:", non_prompt, file=self.logfile)
        print(f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}",file=self.logfile)
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        # transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)
        print(">>>>COMPLETE NOW:",self.to_flush(o),file=self.logfile,flush=True)
        o2 = self.transcript_buffer.complete()
        print("INCOMPLETE:",self.to_flush(o2),file=self.logfile,flush=True)

        # there is a newly confirmed text

        if o and self.buffer_trimming_way == "sentence":  # trim the completed sentences
            if len(self.audio_buffer)/self.SAMPLING_RATE > self.buffer_trimming_sec:  # longer than this
                self.chunk_completed_sentence()

        
        if self.buffer_trimming_way == "segment":
            s = self.buffer_trimming_sec  # trim the completed segments longer than s,
        else:
            s = 30 # if the audio buffer is longer than 30s, trim it
        
        if len(self.audio_buffer)/self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)

            # alternative: on any word
            #l = self.buffer_time_offset + len(self.audio_buffer)/self.SAMPLING_RATE - 10
            # let's find commited word that is less
            #k = len(self.commited)-1
            #while k>0 and self.commited[k][1] > l:
            #    k -= 1
            #t = self.commited[k][1] 
            print(f"chunking segment",file=self.logfile)
            #self.chunk_at(t)

        print(f"len of buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}",file=self.logfile)
        o = self.to_flush(o)
        o2 = self.to_flush(o2)
        if o[0] is not None:
            return (o[0],o[1],o[2], False)
        elif o[1] is None and o[2] == '' \
            and o2[0] is None and o2[1] is None and o2[2] == '':
            return (self.buffer_time_offset, 
                    len(self.audio_buffer)/self.SAMPLING_RATE, 
                    non_prompt,
                    True)
        else:
            return (None, None, "", False)