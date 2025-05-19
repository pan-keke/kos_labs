import pandas as pd
from django.core.management.base import BaseCommand
from data_processor.models import Customer

class Command(BaseCommand):
    help = 'Import customer data from CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to the CSV file')

    def handle(self, *args, **options):
        csv_file = options['csv_file']
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Remove RowNumber as it's not needed
            if 'RowNumber' in df.columns:
                df = df.drop(['RowNumber'], axis=1)
            
            # Create Customer objects
            customers = []
            for _, row in df.iterrows():
                customer = Customer(
                    customer_id=row['CustomerId'],
                    surname=row['Surname'],
                    creditscore=row['CreditScore'],
                    geography=row['Geography'],
                    gender=row['Gender'],
                    age=row['Age'],
                    tenure=row['Tenure'],
                    balance=row['Balance'],
                    numofproducts=row['NumOfProducts'],
                    hascrcard=bool(row['HasCrCard']),
                    isactivemember=bool(row['IsActiveMember']),
                    estimatedsalary=row['EstimatedSalary'],
                    exited=bool(row['Exited'])
                )
                customers.append(customer)
            
            # Bulk create customers
            Customer.objects.bulk_create(customers)
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully imported {len(customers)} customers')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error importing data: {str(e)}')
            ) 