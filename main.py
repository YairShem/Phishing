import urlPred
import emlPred


if __name__ == "__main__":
    print("Hi!")
    print("for eml file press e")
    print("for url address press u")
    ans = input()
    if ans == 'u':
        print("Enter url: ")
        add = input()
        prediction = urlPred.predict(add)
        if prediction == [1]:
            prediction = 'Phishing!!!'
        elif prediction == [0]:
            prediction = 'Not phishing'
        print("The email is: ", prediction)
    elif ans == 'e':
        print("Enter eml file path: ")
        path = input()
        prediction = emlPred.predict(path)
        if prediction == [1]:
            prediction = 'Phishing!!!'
        elif prediction == [0]:
            prediction = 'Not phishing'
        print("The email is: ", prediction)

    else:
        print("Wrong input!!")
        print("Good Bye!")



