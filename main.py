from TrainFile import TrainFile
from TestFile import TestFile

if __name__ == '__main__':
    trainfile = TrainFile(task='喷漆',sheng='大连')
    trainfile.make()
    trainfile.train(thread=8)
    testfile = TestFile(task='喷漆',sheng='大连',starttime='2020-03',endtime='2020-04')
    testfile.make()
    testfile.eval()