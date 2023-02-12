from matplotlib.pyplot import plot, show

velikost = 8


def displaystats(list, starter, seconder):
    CNN = 0
    chatggptminim = 0
    remiza = 0
    for x in list:
        if x == starter:
            CNN += 1
        elif x == seconder:
            chatggptminim += 1
        elif x == '0':
            remiza += 1

    print(
        f"{starter} vyhralo {100 * CNN / len(list)}%, {seconder}vyhral {100 * chatggptminim / len(list)}% a remizovali {100 * remiza / len(list)}%")

    games_won = [1 if x == starter else x for x in list]
    games_won = [-1 if x == seconder else x for x in games_won]
    games_won = [0 if x == "0" else x for x in games_won]
    print(games_won)

    plot(range(len(games_won)), games_won)
    show()
