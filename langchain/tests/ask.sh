#!/bin/bash

# Function to handle the POST request and get request_id
send_post_request() {
    # Read input prompt
    read -p "Enter your prompt: " prompt

    # Send the POST request with the prompt
    response=$(curl -s -X POST http://localhost:8080/ask \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"$prompt\",
            \"permission\": {}
        }")

    # Extract the request_id from the response
    request_id=$(echo "$response" | jq -r '.request_id')


    # Check if request_id was extracted successfully
    if [ -z "$request_id" ]; then
        echo "Error: Could not extract request_id from the response."
        exit 1
    fi

    echo "Request ID: $request_id"
    echo "Now checking status every second until it's completed..."

    # Call the check_status function
    check_status "$request_id"
}

# Function to check the status of the request
check_status() {
    request_id=$1
    status=""

    # Loop to check the status every second
    while [ "$status" != "completed" ]; do
        # Send the GET request to check status
        response=$(curl -s -X GET http://localhost:8080/status/"$request_id")

        # Extract the status from the response using grep
	    status=$(echo "$response" | sed 's/$/\\n/' | tr -d '\n' | sed -e 's/“/"/g' -e 's/”/"/g' | sed '$ s/\\n$//' | jq -r '.status')

        # If status is completed, print the full response
        if [ "$status" = "completed" ]; then
            echo "Status: completed"
            echo "$response" |sed 's/$/\\n/' | tr -d '\n' | sed -e 's/“/"/g' -e 's/”/"/g' | sed '$ s/\\n$//' | jq -r '.response.answer'  # Print the full response JSON
            break
        fi

        # Wait for 1 second before checking again
        sleep 1
    done
}

# Run the script
send_post_request