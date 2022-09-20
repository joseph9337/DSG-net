import argparse
import todo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--todo", default='test', type = str, help="train, test")
    parser.add_argument("--input_shape", default=[256, 3018, 1], type=list)
    parser.add_argument("--label_shape", default=[128, 128, 1], type=list)
    parser.add_argument("--batch_size", default=4, type = int)
    parser.add_argument("--epoch", default=200, type = int)
    parser.add_argument("--model_num", default="1000", type = str)
    parser.add_argument("--drop_out", default="False", type = str)
    parser.add_argument("--DSA_position", default='middle', type=str, help="first, middle, end")
    parser.add_argument("--Model_version", default='D3', type=str, help="D1, D2, D3, D4")
    parser.add_argument("--learning_rate1", default=1e-4, type=float)
    parser.add_argument("--learning_rate2", default=1e-4, type=float)
    ##############################################
    args = parser.parse_args()
    ##############################################
    if args.todo == "train": todo.train(args)
    if args.todo == "test": todo.test(args)

if __name__ == "__main__":
    main()
