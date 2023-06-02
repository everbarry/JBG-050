import argparse

def parser():
    """
    Function that parses the CLI arguments and returns them as dict.
    """
    parser = argparse.ArgumentParser(prog='JBG050',
                                     description='Data Challenge 2 London crime data prediction',
                                     epilog='Made with love by Group 9')
    # add parser arguments
    parser.add_argument('-e', '--epochs', help='specify number of epochs', default=30, type=int)
    parser.add_argument('-l', '--learning-rate', help='specify learning rate', default=3e-4, type=float)
    parser.add_argument('-d', '--weight-decay', help='specify Adam weight decay', default=0.005, type=float)
    parser.add_argument('-a', '--amsgrad', help='if specified Adam is used with amsgrad=True', action='store_true')
    parser.add_argument('-sw', '--sliding-window', help='specify number of past months to use to predict current month', default=24, type=int)
    parser.add_argument('-w', '--n-workers', help='specify number of CPU threads to load data', default=6, type=int)
    parser.add_argument('-s', '--model-summary', help='if specified model summary will be displayed', action='store_true')
    parser.add_argument('-v', '--val', help='if specified load best weights, validate model and display visualizations', action='store_true')
    parser.add_argument('-b', '--wandb', help='if specified logs model training on wandb', action='store_true')
    # return arguments
    return parser.parse_args()
