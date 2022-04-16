import argparse
import json

import predictor

def main(args):
    print(args)

    try:
        output = {}

        svc, nn = predictor.fit_prediction_models()
        output["cuisine"], output["score"] = predictor.find_cuisine(svc, args.ingredient)
        output["closest"] = predictor.find_closest(nn, args.ingredient, args.N)

        print(json.dumps(output, indent=2))
    except BaseException as e:
        print(e.with_traceback())
    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N',required = True, type = int,help='Top N closest meals that are to be revealed')
    parser.add_argument('--ingredient', required = True, type = str, action = "append", help='ingredients are to be inputted')
    
    args = parser.parse_args()   
    
    main(args)