import PySimpleGUI as sg
import time

nl_command_file = "tmp_instr.txt"
nl_real_file = "tmp_instr_real.txt"


def write_command_file(nl_command):
    with open(nl_command_file, "w") as fp:
        fp.write(nl_command)
    print("Command <" + str(nl_command) + "> sent!")

def read_suggest_file():
    with open(nl_real_file, "r") as fp:
        real_nl = fp.read()
    return real_nl or ""

if __name__ == "__main__":
    keep_going = True
    form = sg.FlexForm("Enter the navigation command",
        return_keyboard_events=True,
        default_element_size=(90, 40))
    inputfont = ('Helvetica 30')
    buttonfont = ('Helvetica 20')
    buttoncolor = ("#FFFFFF", "#333333")
    docfont = ('Helvetica 15')

    suggested_text = sg.Text(" ", font=('Helvetica 15'))
    input_field = sg.Input(font=inputfont)
    layout = [
        [suggested_text],
        [sg.Text("    Enter:", font=docfont), sg.Text("Execute command", font=docfont)],
        [sg.Text("    Left:  ", font=docfont), sg.Text("Reset the drone to starting position", font=docfont)],
        [sg.Text("    Right:", font=docfont), sg.Text("Go to the next environment", font=docfont)],
        [input_field],
        [sg.ReadFormButton('OK', font=buttonfont, button_color=buttoncolor),
        sg.ReadFormButton('Next',  font=buttonfont, button_color=buttoncolor),
        sg.ReadFormButton("Reset",  font=buttonfont, button_color=buttoncolor),
        sg.ReadFormButton('Clear Text',  font=buttonfont, button_color=buttoncolor)]
    ]

    while keep_going:
        real_nl_cmd = read_suggest_file()
        print("Suggested CMD", real_nl_cmd)
        layout[0] = [sg.Text("    Down:  Use suggested: " + real_nl_cmd, font=('Helvetica 15'))]
        form.Layout(layout)

        #suggested_text.Update()

        # ---===--- Loop taking in user input --- #
        while True:
            button, values = form.ReadNonBlocking()
            #print("Got: ", button, values)
            nl_command = values[0] if values else ""

            if button in ["Next", "Right:114"]:
                nl_command = "CMD: Next"
                input_field.Update("")
                break

            elif button in ["OK", "\r"] and nl_command:
                print("Executing: " + str(nl_command))
                break

            elif button in ["Reset", "Left:113"] or (button and ord(button[0]) == 0x001B):
                print("Reseting")
                nl_command = "CMD: Reset"
                #input_field.Update("")
                break

            elif button in ["Clear Text"]:
                print("Clearing field")
                input_field.Update("")

            elif button in ["Down:116"]:
                input_field.Update(real_nl_cmd)

            elif button is None and values is None:
                keep_going = False
                break
            time.sleep(0.05)

        write_command_file(nl_command)
