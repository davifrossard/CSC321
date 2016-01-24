from get_data_file import *

def fetch_sets(source, actors, train=100, validation=10, test=10):
    x_train = []
    x_validation = []
    x_test = []

    t_train = []
    t_validation = []
    t_test = []

    for actor in actors:
        num_points, _, faces, _ = fetch_data(source, [actor], train+validation+test)
        if num_points >= train+validation+test:
            for i in range(0,train):
                x_train.append(faces[i])
                t_train.append(actor)
            for i in range(train,train+validation):
                x_validation.append(faces[i])
                t_validation.append(actor)
            for i in range(train+validation,train+validation+test):
                x_test.append(faces[i])
                t_test.append(actor)
        else:
            print "Not enough data for actor "+actor

    return x_train, t_train, x_validation, t_validation, x_test, t_test
