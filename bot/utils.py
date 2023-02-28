from matplotlib.pyplot import plot, show, subplots



def displaystats(game_record,games_len, starter, seconder):
    starter_wins = 0
    seconder_wins = 0
    remiza = 0
    for x in game_record:
        if x == starter:
            starter_wins += 1
        elif x == seconder:
            seconder_wins += 1
        elif x == '0':
            remiza += 1

    print(
        f"{starter} vyhralo {100 * starter_wins / len(game_record)}%, {seconder}vyhral {100 * seconder_wins / len(game_record)}% a remizovali {100 * remiza / len(game_record)}%")

    games_won = [1 if x == starter else x for x in game_record]
    games_won = [-1 if x == seconder else x for x in games_won]
    games_won = [0 if x == "0" else x for x in games_won]
    print(games_won)

    figure, axis = subplots(2, 1)
    axis[0].plot(range(len(games_won)), games_won)
    axis[0].set_title("Winnings")

    # For Cosine Function
    axis[1].plot(range(len(games_len)),games_len)
    axis[1].set_title("Game length")
    show()
