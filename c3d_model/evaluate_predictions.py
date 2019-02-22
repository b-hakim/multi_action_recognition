def evaluate_prediction(file_path = 'predict_ret.txt'):
    correct = 0
    incorrect = 0
    toread = 0

    with open(file_path) as f:
        toread = len(list(f))

    with open(file_path) as predictions:
        #print len(list(predictions))

        for i, l in enumerate(predictions):
            if i == toread:
                break

            rec = [float(val) for val in l.split(',')]

            if rec[0] == rec[2]:
                correct+=1
            else:
                incorrect+=1

    print(correct/float(correct+incorrect))
    print(correct)
    print(incorrect)

    return correct/float(correct+incorrect), correct, incorrect

if __name__ == '__main__':
    evaluate_prediction()