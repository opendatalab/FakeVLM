import torch
import os
import argparse
from dataclasses import dataclass, field
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPImageProcessor
import pdb
import json
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from vllm import LLM, SamplingParams
import pdb
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def parse_args():
    parser = argparse.ArgumentParser(description="FakeVLM Model Testing")

    # Model-specific settings
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--data_base_test", default="", type=str)
    parser.add_argument("--test_json_file", default="", type=str)  
    parser.add_argument("--output_path", default="", type=str)  
    return parser.parse_args()

class legion_cls_dataset(Dataset):
    def __init__(self, args, train=True):
        super().__init__()
        self.args = args
        self.train = train
        if train == True:
            with open(args.train_json_file, 'r') as f:
                self.data = json.load(f)
        elif train == False:
            with open(args.test_json_file, 'r') as f:
                self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.train == True:
            img_path = os.path.join(self.args.data_base_train, self.data[idx]['image'])
        else:
            img_path = os.path.join(self.args.data_base_test, self.data[idx]['image'])


        label = self.data[idx]['label']

        cate = self.data[idx]['cate']

        # torch.Size([n, 3, 448, 448]),  int, int, str, str
        return [self.data[idx]['conversations'][0]['value']], [label], [img_path], [cate]




# change this function to our own to support custom behaviors
def load_model(args):
    print("Loading model...")
    llm = LLM(model=args.model_path,
        dtype="float16",
        max_model_len=800,
        tensor_parallel_size=torch.cuda.device_count()
    )
    print("Successfully loaded model from:", args.model_path)
    return llm

def calculate_results_acc(results):
    acc_results = {}
    
    for cate in results:
        data = results[cate]
        
        right_real = data['right']['right_real']
        right_fake = data['right']['right_fake']
        wrong_real = data['wrong']['wrong_real']
        wrong_fake = data['wrong']['wrong_fake']
        
        total_real = right_real + wrong_real 
        total_fake = right_fake + wrong_fake 
        total = total_real + total_fake      
        
        acc_total = (right_real + right_fake) / total if total != 0 else 0
        acc_real = right_real / total_real if total_real != 0 else 0
        acc_fake = right_fake / total_fake if total_fake != 0 else 0
        
        acc_results[cate] = {
            'total_samples': total,
            'total_accuracy': round(acc_total, 4),
            'real_accuracy': round(acc_real, 4),
            'fake_accuracy': round(acc_fake, 4),
            'overall': {
                'right_real': right_real,
                'wrong_real': wrong_real,
                'right_fake': right_fake,
                'wrong_fake': wrong_fake, 
            }
        }
    
    global_stats = {
        'total_right': sum(r['right']['right_real'] + r['right']['right_fake'] for r in results.values()),
        'total_wrong': sum(r['wrong']['wrong_real'] + r['wrong']['wrong_fake'] for r in results.values())
    }
    global_stats['global_accuracy'] = global_stats['total_right'] / (global_stats['total_right'] + global_stats['total_wrong'])
    
    return {
        'category_acc': acc_results,
        'global_stats': global_stats
    }



def validate(args, model, cls_test_dataloader):
    results = {}
    output_result = []
    sampling_params = SamplingParams(
        max_tokens=4096, 
        temperature=0,             
    )

    with torch.no_grad():
        for questions, labels, imgs, cates in tqdm(cls_test_dataloader):
            inputs = []  

            for question, img in zip(questions[0], imgs[0]):
                inputs.append({
                    "prompt": question,
                    "multi_modal_data": {"image": Image.open(img)},
                })
            # pdb.set_trace()
            outputs = model.generate(inputs, sampling_params=sampling_params)

            pred_cls = []

            for i, output in enumerate(outputs):
                # pdb.set_trace()
                
                # #save result
                output_result.append({'id':imgs[0][i], 'caption':output.outputs[0].text})


                response = output.outputs[0].text
                if 'real' in response.split('.')[0].lower():
                    pred_cls.append(1)
                elif 'fake' in response.split('.')[0].lower():
                    pred_cls.append(0)
                else:
                    try:
                        if 'real' in response.split('.')[1].lower():
                            pred_cls.append(1)
                        elif 'fake' in response.split('.')[1].lower():
                            pred_cls.append(0)
                        else:
                            print(f"no fake or real in reponse:{response}")
                            pred_cls.append(random.choice([0, 1]))
                    except:
                            print(f"no fake or real in reponse:{response}")
                            pred_cls.append(random.choice([0, 1]))                        

            for label, pred, cate in zip(labels[0].tolist(), pred_cls, cates[0]):
                if cate not in results:
                    results[cate] = {'right':{'right_fake':0, 'right_real':0}, 'wrong':{'wrong_fake':0, 'wrong_real':0}}
                if label == pred:
                    if label == 1:
                        results[cate]['right']['right_real'] += 1
                    else:
                        results[cate]['right']['right_fake'] += 1                        
                else:
                    if label == 1:
                        results[cate]['wrong']['wrong_real'] += 1
                    else:
                        results[cate]['wrong']['wrong_fake'] += 1         
    
    # save result 
    os.makedirs("results", exist_ok=True)      
    with open(args.output_path, "w") as file:
        json.dump(output_result, file, indent=2)
 
    acc = calculate_results_acc(results)
    print(acc)

        
    
def main():
    args = parse_args()
    model = load_model(args)
    cls_test_dataset = legion_cls_dataset(args, train=False)
    cls_test_dataloader = DataLoader(
        cls_test_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    validate(args, model, cls_test_dataloader)

    


if __name__ == "__main__":
    main()
    
    