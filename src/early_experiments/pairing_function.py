def pair(x, y):
    return (x + y) * (x + y + 1) // 2 + y

def unpair(z):
    w = (8 * z + 1) ** 0.5
    t = (w - 1) // 2
    y = int(z - (t * (t + 1)) // 2)
    x = int(t - y)
    return x, y


if __name__ == '__main__':
    # x, y = 2, 2
    # z = pair(x, y)
    # print(f"Pairing: ({x}, {y}) -> {z}")
    
    x_unpair, y_unpair = unpair(50)
    print(f"Unpairing: {98} -> ({x_unpair}, {y_unpair})")
