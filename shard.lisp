;;;; shard.lisp — embed a chunk range (env: SHARD_START, SHARD_END, SHARD_THREADS).
;;;; Reads chunks.jsonl, writes out/{index.bin,meta.tsv,texts.tsv,index.model}.
(dolist (f '("config" "json" "st" "tokenizer" "llama" "bert" "gemma" "model" "embed"))
  (load (merge-pathnames (concatenate 'string f ".fasl") *load-pathname*)))
(setf embed::*chunks* #P"chunks.jsonl"
      embed::*index* #P"out/index.bin"
      embed::*meta* #P"out/meta.tsv"
      embed::*texts* #P"out/texts.tsv"
      embed::*model-marker* #P"out/index.model"
      embed::*model* (cfg:env "KB_MODEL" "model/model.safetensors"))
(ensure-directories-exist #P"out/")
(embed:run :start (parse-integer (cfg:env "SHARD_START" "0"))
           :end (parse-integer (cfg:env "SHARD_END" "0"))
           :threads (parse-integer (cfg:env "SHARD_THREADS" "4")))
