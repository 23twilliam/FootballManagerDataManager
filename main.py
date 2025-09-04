import utils.preprocessing
from visualisation import plotter
import analysis
def main():
    #fileLocation = str(input('Enter file location: '))
    fileLocation = "E:/pythonProject/PlayerInfo/2044PlayerInfo.csv"
    position = int(input('Enter position: \n '
                         '1. Keeper \n '
                         '2. Halfback \n '
                         '3. Fullback \n '
                         '4. Wingback \n '
                         '5. Defensive Midfielder \n '
                         '6. Midfielder \n '
                         '7. Attacking Midfielder \n '
                         '8. Winger \n '
                         '9. Striker \n '))

    valMax = int(input("What is the maximum value you would consider"))
    df = utils.preprocessing.preprocess(fileLocation)

    match position:
        case 1:
            avg = analysis.keeper(df, valMax)
        case 2:
            avg = analysis.defender(df, valMax)
        case 3:
            avg = analysis.fullback(df, valMax)
        case 4:
            avg = analysis.wingback(df, valMax)
        case 5:
            avg = analysis.defensive_midfielder(df, valMax)
        case 6:
            avg = analysis.central_midfielder(df, valMax)
        case 7:
            avg = analysis.attacking_midfielder(df, valMax)
        case 8:
            avg = analysis.winger(df, valMax)
        case 9:
            avg = analysis.striker(df, valMax)
        case _:
            avg = None  # Default case if position doesn't match any
    plotter.plotting(avg, df, valMax, position)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
