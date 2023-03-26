import argparse

from tempTrain import trainAPI
from net.PSP import build_PSPnet
from yamlparser import read_config

def main():
    # parser = argparse.ArgumentParser(description="Semantic Segmentation Model")
    
    # parser.add_argument("-train", type = str, help = "config file", required = False)
    
    # args = parser.parse_args()
    
    # if args.train is not None:
    #     kwargs = read_config(args.train)
    #     train(kwargs)
        
    # else:
    #     print("No Train")
    # kwargs = read_config("config.yaml")
    # Tr = trainAPI(**kwargs)

    # Tr.run()

    m = build_PSPnet((256, 256, 3), 150, 512, (1,2,3,6), 1)
    m.summary()
        

if __name__ == "__main__":
    main()
    
# python main.py -train config.yaml