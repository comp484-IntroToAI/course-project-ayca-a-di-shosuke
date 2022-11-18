# https://www.geeksforgeeks.org/longest-common-subsequence-dp-4/

class Rouge():
    def lcs(self, X, Y):
        # find the length of the strings
        m = len(X)
        n = len(Y)

        # declaring the array for storing the dp values
        # L = [[None] * (n + 1) for i in range(m + 1)]
        L = [[-1 for i in range(n + 1)] for j in range(m + 1)]

        """Following steps build L[m+1][n+1] in bottom up fashion 
        Note: L[i][j] contains length of LCS of X[0..i-1] 
        and Y[0..j-1]"""
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i - 1] == Y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

                    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
        return L[m][n]
        # end of function lcs

    def metrics(self, X, Y):
        lcs = rouge.lcs(X, Y)
        print("Length of LCS is ", lcs)
        precision = lcs/len(Y)
        recall = lcs/len(X)
        f1 = (2 * precision * recall) / (precision + recall)
        print("Precision: " + str("%.2f" % precision))
        print("Recall: " + str("%.2f" % recall))
        print("F1-Score: " + str("%.2f" % f1))
        return precision, recall, f1

if __name__ == "__main__":
    rouge = Rouge()
    # X = "AGGTAXB"
    # Y = "GXTXAYB"
    # X = ["A", "G", "G", "T", "A", "X", "B"]
    # Y = ["G", "X", "T", "X", "A", "Y", "B"]
    X = ["The", "cat", "is", "on", "the", "mat", "."]
    Y = ["The", "cat", "and", "the", "dog", "."]
    # X = ["Hello, my name is Pooh.", "I'm from California.", "Nice to meet you."]
    # Y = ["Hello, my name is Pooh.", "Nice to meet you."]
    result = rouge.metrics(X, Y)
