# split: 40 for train, 10 for validation per class

import csv


def read_whole_data_set():
    data = []
    with open('data/iris.data', 'r') as csvfile:
        iris_reader = csv.reader(csvfile)
        for row in iris_reader:
            features = row[:4]
            if 0 == len(features):
                break

            features = [float(x) for x in features]
            label = row[4]

            item = {'vec': features, 'label': label}
            data.append(item)
    return data


def write_split(data):
    train_writer = csv.writer(open('data/iris.data.train', 'w'))
    test_writer = csv.writer(open('data/iris.data.test', 'w'))

    it = 0

    classes_count = 3
    for _ in range(classes_count):
        train_samples_count = 40
        for _ in range(train_samples_count):
            row = data[it]['vec'] + [data[it]['label']]
            train_writer.writerow(row)
            it += 1

        test_samples_count = 10
        for _ in range(test_samples_count):
            row = data[it]['vec'] + [data[it]['label']]
            test_writer.writerow(row)
            it += 1


def main():
    data = read_whole_data_set()
    write_split(data=data)

    # pprint(data)

if __name__ == '__main__':
    main()
