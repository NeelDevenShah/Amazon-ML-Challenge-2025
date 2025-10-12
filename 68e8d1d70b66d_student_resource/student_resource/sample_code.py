import os
import random
import pandas as pd

def predictor(sample_id, catalog_content, image_link):
    '''
    Call your model/approach here
    
    Parameters:
    - sample_id: Unique identifier for the sample
    - catalog_content: Text containing product title and description
    - image_link: URL to product image
    
    Returns:
    - price: Predicted price as a float
    '''
    # TODO: Implement your price prediction logic here
    # This is just a dummy implementation
    
    # Generate random price between 5 and 500
    return round(random.uniform(5.0, 500.0), 2)

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    
    # Read test data
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Apply predictor function to each row
    test['price'] = test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row['image_link']), 
        axis=1
    )
    
    # Select only required columns for output
    output_df = test[['sample_id', 'price']]
    
    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Sample predictions:\n{output_df.head()}")
