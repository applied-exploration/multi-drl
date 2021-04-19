
def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def unique(a):
    return list(set(a))

def mean(lst):
    return sum(lst) / len(lst)