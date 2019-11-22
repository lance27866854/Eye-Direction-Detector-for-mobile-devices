
def gradient_generator(G1, G2, G3):
    # G: [height, width, 3]
    height = G2.shape[0]
    width = G2.shape[1]

    candidate_list = []
    for i in range(2, height-2):
        for j in range(2, width-2):
            G1_max = np.max(G1[i-2:i+3, j-2:j+3])
            G2_max = np.max(G2[i-2:i+3, j-2:j+3])
            G3_max = np.max(G3[i-2:i+3, j-2:j+3])
            G1_min = np.min(G1[i-2:i+3, j-2:j+3])
            G2_min = np.min(G2[i-2:i+3, j-2:j+3])
            G3_min = np.min(G3[i-2:i+3, j-2:j+3])
            maximum = np.max([G1_max, G2_max, G3_max])
            minimum = np.min([G1_min, G2_min, G3_min])
            if np.max(G2[i][j]) == maximum or np.min(G2[i][j]) == minimum:
                Gr = (sum(pow(pow(G2[i+1][j]-G2[i-1][j], 2)+pow(G2[i][j+1]-G2[i-1][j-1], 2), 0.5)), i, j)
                candidate_list.append(Gr)
                
    candidate_list = sorted(candidate_list, key = lambda x : x[0], reverse=True)
    return candidate_list

'''
# TODO:
- try gray-scale images.

'''