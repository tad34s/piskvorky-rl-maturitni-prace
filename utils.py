from matplotlib.pyplot import plot, show

velikost = 8


def displaystats(list, starter, seconder):
    starter_wins = 0
    seconder_wins = 0
    remiza = 0
    for x in list:
        if x == starter:
            starter_wins += 1
        elif x == seconder:
            seconder_wins += 1
        elif x == '0':
            remiza += 1

    print(
        f"{starter} vyhralo {100 * starter_wins / len(list)}%, {seconder}vyhral {100 * seconder_wins / len(list)}% a remizovali {100 * remiza / len(list)}%")

    games_won = [1 if x == starter else x for x in list]
    games_won = [-1 if x == seconder else x for x in games_won]
    games_won = [0 if x == "0" else x for x in games_won]
    print(games_won)

    plot(range(len(games_won)), games_won)
    show()
