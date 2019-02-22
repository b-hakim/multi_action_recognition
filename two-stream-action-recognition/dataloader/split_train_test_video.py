import os, pickle


class Office_Actions_splitter():
    def __init__(self, trainfile, testfile, is_label_zero_based):
        self.train_path, self.test_path = trainfile, testfile
        if is_label_zero_based:
            self.label_offset = 1
        else:
            self.label_offset = 0

    def get_action_index(self):
        self.action_label={}
        with open(os.path.dirname(self.train_path)+'/class_index.txt') as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        max_cls_ind=-1
        for line in content:
            if line.find(',') != -1:
                label,action = line.split(',')
            else:
                label,action = line.split()

            label = int(label) + self.label_offset

            #print label,action
            if action not in self.action_label.keys():
                self.action_label[action]=label
                self.action_label[label] = action
                if label > max_cls_ind:
                    max_cls_ind = label
        return max_cls_ind

    def split_video(self):
        max_cls_ind = self.get_action_index()

        train_video = self.file2_dic(self.train_path)
        test_video = self.file2_dic(self.test_path)

        print '==> (Training video, Validation video):(', len(train_video),len(test_video),')'

        return train_video, test_video, max_cls_ind

    def file2_dic(self,fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        # import random
        # random.shuffle(content)
        # content = content[:10]
        dic={}
        for line in content:
            #print line
            # video = line.split('/',1)[1].split(' ',1)[0]
            if line.find(',') != -1:
                key = line.split(',')[0] # video.split('.',1)[0]
                label = line.split(',')[1]
            else:
                key = line.split()[0]  # video.split('.',1)[0]
                label = line.split()[1]
            dic[key.replace(".mp4", "")] = int(label)+self.label_offset
            #print key,label
        return dic


if __name__ == '__main__':

    path = '../UCF_list/'
    split = '01'
    splitter = Office_Actions_splitter(path=path)
    train_video,test_video = splitter.split_video()
    print len(train_video),len(test_video)