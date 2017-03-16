from copy import deepcopy
def flatten(iterable):
    results = []
    for i in iterable:
        #if isinstance(i, collections.Iterable) and not isinstance(i, basestring):
        if isinstance(i, list) or isinstance(i, tuple):
            results.extend(flatten(i))
        else:
            results.append(i)
    return results

def unflatten(fllatened,unfllatened):
    return _unflatten(deepcopy(fllatened), unfllatened)

def _unflatten(fllatened,unfllatened):
    result = []
    for i in unfllatened:
        if isinstance(i, list) or isinstance(i, tuple):
            result.append(_unflatten(fllatened,i))
        else:
            result.append(fllatened[0])
            del fllatened[0]

    return type(unfllatened)(result)


if __name__ == '__main__':
    a = [(1,2),3,(4,5,(6,7),[8])]
    b = flatten(a)
    c = unflatten(b,a)

    print a
    print b
    print c

