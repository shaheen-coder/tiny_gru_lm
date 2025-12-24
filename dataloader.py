import sentencepiece as spm 

class DataLoader:

    def __init__(self,filename : str, model_filename : str ) -> None :
        self.filename = filename
        self.model_filename = model_filename
        self.dataset_file = None 
        self.tokenizer = None 
    
    def load(self) -> bool :

        try : 
            self.dataset_file = open(self.filename,'r')
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(self.model_filename)
            return True 
        except Exception as e:
            print(f'error : {e}')
            return False 

    def get(self) -> list[int]:
        if(self.load()):
            ids : list[int] = []
            lines = self.dataset_file.readlines()
            for line in lines:
                tokens = self.tokenizer.encode(line)
                ids.append(tokens) 
            return ids
    def get_with_pad(self) -> list[list[int]]:
        datas: list[list[int]] = self.get()
        max_len = 50
        
        pad_token: int = self.tokenizer.encode('<pad>')[0] 
        
        final_tok: list[list[int]] = []
        
        for data in datas:
            curr_len = len(data)
            if curr_len < max_len:
                # Efficiently extend using list multiplication
                padding = [pad_token] * (max_len - curr_len)
                final_tok.append(data + padding)
            else:
                # Slice to max_len
                final_tok.append(data[:max_len])
            
        return final_tok

    def write(self,filename : str ) -> None :
        with open(filename,'w') as file:
            datas : list[int] = self.get()
            for data in datas:
                print(data)
                file.write(','.join(map(str,data)))
                file.write('\n')

if __name__ == '__main__':
    # testing DataLoader
    dl = DataLoader('out.txt','tokenizer.model')
    ids = dl.get_with_pad()
    if ids:
        total_length = sum(len(row) for row in ids)
        avg_length = total_length / len(ids)
        print(f"Average ID length: {avg_length:.2f}")
    else:
        print("Dataset is empty.")
