import os
import time
import logging

from agent import OSXProcAgent
# import torch.multiprocessing

logger = logging.getLogger(__name__)

def main():
    osx_proc_agent = OSXProcAgent()
    osx_proc_agent.run()
    

if __name__=="__main__":
    logging.basicConfig(level=logging.WARNING)
    main()


