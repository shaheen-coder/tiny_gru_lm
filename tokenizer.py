import sentencepiece as spm

IS_TRAIN : bool = True 
FILENAME : str = "out.txt"

user_defined_symbols : list[str] = ['<ai>', '<prompt>', '<end>','<pad>']

if IS_TRAIN:
    # Train the model
    spm.SentencePieceTrainer.Train(
        input=FILENAME,
        model_prefix='tokenizer',
        vocab_size=2048,
        character_coverage=1.0,
        model_type='bpe',
        user_defined_symbols=user_defined_symbols,
    )

    # Load the model
    sp = spm.SentencePieceProcessor()
    sp.Load('tokenizer.model')
else:
    
    # Tokenize text
    while True:
        text = input("Enter your text : ")
        if text == 'q' : break
        tokens = sp.EncodeAsPieces(text)
        ids = sp.encode(text)
        print("encode as text : ",tokens)
        print("encode as int : ",ids,f"\n len ids : {len(ids)} and token : {len(text.split(' '))} ")

