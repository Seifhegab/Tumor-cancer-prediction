class Voting:
    def __init__(self, Y_pred_log, Y_predtree, Y_predsv):
        self.Y_pred_log = Y_pred_log
        self.Y_predtree = Y_predtree
        self.Y_predsv = Y_predsv

    def vote(self):
        # Voting the prediction
        counter1 = 0
        counter0 = 0
        for i in range(len(self.Y_pred_log)):

            if self.Y_pred_log[i] == 1:
                counter1 = counter1 + 1
            else:
                counter0 = counter0 + 1

            if self.Y_predtree[i] == 1:
                counter1 = counter1 + 1
            else:
                counter0 = counter0 + 1

            if self.Y_predsv[i] == 1:
                counter1 = counter1 + 1
            else:
                counter0 = counter0 + 1

            if counter1 > counter0:
                print("patient num ", i + 1, " will have tumor")
            else:
                print("patient num ", i + 1, " will not have tumor")
            counter1 = 0
            counter0 = 0
        print('\n')
