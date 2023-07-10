from arguments import parser 
from pathlib import Path
import os 
import ruamel.yaml as yaml


if __name__ == '__main__':
    config = parser()
    args = config.args
    config['output_dir'] = args.output_dir 
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if config.TASK in ['VQA','Grounding']:
        args.result_dir = os.path.join(args.output_dir, 'result')
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))   
    
    # Train 
    if not args['evaluate']: # args['evaluate] : only Evaluation 
        main = __import__(f'{config.TASK}').__dict__['main']
        main(args, config)    
        
    # Evaluation Image 
    args['checkpoint'] = os.path.join(args['output_dir'],'checkpoint_best.pth')    
    eval_image = __import__(f'{config.TASK}').__dict__['eval_image']
    eval_image(args, config)
    
    # Evaluation Text 
    eval_text = __import__(f'{config.TASK}').__dict__['eval_text']
    eval_text(args, config)
    