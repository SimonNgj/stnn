
TRAIN_FILES = ['../data/myMCH/',  # 0
               '../data/myMCH3/', # 1
               '../data/suture/', # 2
               '../data/needle/', # 3
               '../data/knot/',   # 4
               ]

EVAL_FILES = ['../data/myMHC/',   # 0
               '../data/myMHC3/', # 1
               '../data/suture/', # 2
               '../data/needle/', # 3
               '../data/knot/',   # 4
              ]

MAX_NB_VARIABLES = [6,  #0
                    6,  #1
                    76, #2
                    76, #3
                    76, #4
                    ]

MAX_TIMESTEPS = [958,  #0
                 958,  #1 
                 9013, #2 
                 5357, #3
                 3854, #4  
                 ]

NB_CLASSES = [2, #0
              3, #1
              3, #2
              3, #3
              3, #4
              ]