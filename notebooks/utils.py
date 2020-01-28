from fastai import *
from fastai.text import *
from sklearn.metrics import roc_auc_score
from rdkit import Chem
import numpy as np

def randomize_smiles(smiles, N_rounds):
    smiles_random = [None] * N_rounds
    m = Chem.MolFromSmiles(smiles)
    N = m.GetNumAtoms()
    
    for i in range(N_rounds):
        m_renum = Chem.RenumberAtoms(m, np.random.permutation(N).tolist())
        smiles_random[i] = Chem.MolToSmiles(m_renum, canonical=False, isomericSmiles=True, kekuleSmiles=False)
        
    return list(set(smiles_random))

def smiles_augmentation(df, N_rounds):   
    SMILES = df['SMILES'].to_list()
    canonical = df['canonical'].to_list()

    SMILES_aug = []    
    for smiles in SMILES:
        smiles_random = randomize_smiles(smiles, N_rounds)
        SMILES_aug += smiles_random
            
    canonical_aug = ['no'] * len(SMILES_aug)

    #merge with original data
    df = pd.DataFrame.from_dict({'SMILES' : SMILES + SMILES_aug,
                                 'canonical' : canonical + canonical_aug})
    
    #shuffle the data   
    return df.loc[np.random.permutation(df.index), :]

# Don't include the defalut specific token of fastai, only keep the padding token
BOS,EOS,FLD,UNK,PAD = 'xxbos','xxeos','xxfld','xxunk','xxpad'
TK_MAJ,TK_UP,TK_REP,TK_WREP = 'xxmaj','xxup','xxrep','xxwrep'
defaults.text_spec_tok = [PAD]

# Special tokens that must be kept together
special_tokens = [ '[BOS]', # beginning of sequence
                   '[C@H]', '[C@@H]','[C@]', '[C@@]','[C-]','[C+]', '[c-]', '[c+]','[cH-]', # carbon chiralities
                   '[nH]', '[N+]', '[N-]', '[n+]', '[n-]' '[NH+]', '[NH2+]', # nitrogen
                   '[O-]', '[O+]', # oxygen
                   '[S+]', '[s+]', '[S-]', '[SH]', # sulfur
                   '[B-]','[BH2-]', '[BH3-]','[b-]', # boron
                   '[PH]','[P+]', # phosphorous 
                   '[I+]', # halogen
                   '[Si]','[SiH2]', '[Se]','[SeH]', '[se]', '[Se+]', '[se+]','[te]','[te+]', '[Te]' # metallic elements
                 ]

# Elements of our interest that have two letter symbols
merge_elements = ['Br', 'Cl']

class MolTokenizer(BaseTokenizer):
    def __init__(self, lang = 'en', special_tokens = special_tokens, merge_elements = merge_elements):
        super().__init__(lang = lang)
        self.__special_tokens = special_tokens
        self.__merge_elements = merge_elements if merge_elements is not None else []
        
    def tokenizer(self, smiles):
        # add specific token '[BOS]' to represetences the start of SMILES
        regex = '(\[[^\[\]]{1,10}\])'
        parts = re.split(regex, smiles)
        tokens = []
       
        if self.__special_tokens:
            for part in parts:
                if part.startswith('['):
                    tokens += [part] if part in self.__special_tokens else ['[UNK]'] 
                else:
                    tokens += list(part)
        else:
            for part in parts:
                tokens += [part] if part.startswith('[') else list(part)                

        # merge two letter element tokens
        for element in self.__merge_elements:
            MolTokenizer.patch_double(tokens, element)
        
        return ['[BOS]'] + tokens
    
    @staticmethod
    def patch_double(tokens, pair):
        first, second = list(pair)
        while True:
            try:
                ix = tokens.index(second)

                if tokens[ix - 1] == first:
                    tokens[ix - 1] = first + second
                    del tokens[ix]
            except ValueError:
                return tokens
        
    def add_special_cases(self, tokens):
        pass

def auroc_score(input, target):
    input, target = input.cpu().numpy()[:,1], target.cpu().numpy()
    return roc_auc_score(target, input)

class AUROC(Callback):
    _order = -20 #Needs to run before the recorder

    def __init__(self, learn, **kwargs): self.learn = learn
    def on_train_begin(self, **kwargs): self.learn.recorder.add_metric_names(['AUROC'])
    def on_epoch_begin(self, **kwargs): self.output, self.target = [], []
    
    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            self.output.append(last_output)
            self.target.append(last_target)
                
    def on_epoch_end(self, last_metrics, **kwargs):
        if len(self.output) > 0:
            output = torch.cat(self.output)
            target = torch.cat(self.target)
            preds = F.softmax(output, dim=1)
            metric = auroc_score(preds, target)
            return add_metrics(last_metrics, [metric])



def test_get_scores(learn, ret=False):
    preds = learn.get_preds(ordered=True)
    print(f'Testing {len(preds[0])} molecues')
    p = torch.argmax(preds[0], dim=1)
    y = preds[1]
    tp = ((p + y) == 2).sum().item()
    tn = ((p + y) == 0).sum().item()
    fp = (p > y).sum().item()
    fn = (p < y).sum().item()
    cc = (float(tp)*tn - fp*fn) / np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    
    print(f'Accuracy: {(tp+tn)/len(y):.3f}')
    print(f'False Positives: {fp/len(y):.3f}')
    print(f'False Negatives: {fn/len(y):.3f}')
    print(f'Recall: {tp / (tp + fn):.3f}')
    print(f'Precision: {tp / (tp + fp):.3f}')
    print(f'Sensitivity: {tp / (tp + fn):.3f}')
    print(f'Specificity: {tn / (tn + fp):.3f}')
    print(f'MCC: {cc:.3f}')
    print(f'ROCAUC: {roc_auc_score(y,preds[0][:,1]):.3f}')

    if ret:
        return preds




