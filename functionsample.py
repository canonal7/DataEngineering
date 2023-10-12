def process_data(event, context):
    """Triggered by a Pub/Sub message."""
    import base64

    print("""This function was triggered by messageId {} published at {} with attributes:""".format(context.event_id, context.timestamp))

    if 'data' in event:
        name = base64.b64decode(event['data']).decode('utf-8')
        print('Processed file: {}'.format(name))
    else:
        print('No data found!')
