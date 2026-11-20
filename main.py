
from analyzer import (
    load_data,
    analyze_data,
    train_model,
    evaluate_model,
    predict_ticket
)


def main():
    print("=" * 45)
    print(" CUSTOMER SUPPORT TICKET CLASSIFIER")
    print("=" * 45)

    # Load dataset
    df = load_data()

    # Analyze dataset
    analyze_data(df)

    # Train model
    print("\nTraining machine learning model...")
    model, X_test, y_test = train_model(df)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Predict new tickets
    print("\nTicket Classification")
    print("---------------------")

    while True:
        ticket = input(
            "\nEnter customer complaint (or 'exit'): "
        ).strip()

        if ticket.lower() == "exit":
            print("Classifier closed.")
            break

        if not ticket:
            print("Please enter a complaint.")
            continue

        category = predict_ticket(model, ticket)

        print("Predicted Category:", category)


if __name__ == "__main__":
    main()
