from main import main

if __name__ == "__main__":
    days_list = [30, 60, 90]
    for d in days_list:
        print("\n==============================")
        print(f"Running main with days_to_maturity={d}")
        print("==============================\n")
        main(days_to_maturity=d, show_plots=False)
