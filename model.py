import numpy as np
import json


class VelocityModel():
    """Model to predict imparted fragment velocity."""
    def __init__(self, config: str = None, intercept: float = None, 
                 coefficients: list = None, check_results=True):
        """Initialize the model with a config or intercept and coefficients.
        Args:
            config: Path to configuration file.
            intercept: Intercept value of the model.
            coefficients: List of coefficients for the model.
            check_results: If True, the model will check the results for anomalies 
            apearing from too low am and diameter combinationsand correct them if necessary.
        """
        if config:
            self.config = config
            self.create_model_from_config()
        elif intercept and coefficients:
            self.intercept = intercept
            self.coefficients = np.array(coefficients)
            assert len(self.coefficients) == 10, "Model must have 10 coefficients"
        else:
            raise ValueError("Either config or coefficients and intercept must be provided")
        self.check_result= check_results
    
    def create_model_from_config(self):
        """Create the model based on configuration file."""
        with open(self.config) as f:
            self.config = json.load(f)
        self.intercept = self.config['intercept']
        self.coefficients = np.array(self.config['coefficients'])
        self.limits = self.config['limits']
        self.log_transformers = self.config['log_transform']

    def calculate(self, x):
        """Apply regression model to matrix x."""
        return self.intercept + np.dot(x, self.coefficients)
    
    def transform(self, x):
        """Transform input values by applying log and interaction terms."""
        transformed = np.log(x + self.log_transformers[:4])
        # Creating interaction terms as a matrix
        interactions = np.array([transformed[:, i] * transformed[:, j]
                                  for i in range(4) for j in range(i + 1, 4)]).T
        return np.hstack([transformed, interactions])
    
    def apply_limits(self, x):
        """Apply limits to the input matrix x."""
        if self.limits:
            for i in range(4):
                x[:, i] = np.clip(x[:, i], self.limits[i][0], self.limits[i][1])
        return x
    
    def predict(self, inputs):
        """Predict the velocity for each input row in inputs."""
        inputs = np.array(inputs)
        inputs[:, 0] = 1 / (0.005 * inputs[:, 0])  # Convert am to density
        inputs = self.apply_limits(inputs)
        transformed_inputs = self.transform(inputs)
        velocities = self.calculate(transformed_inputs)
        velocities = np.where(velocities < 0, 0, velocities)  # Validate results
        if self.check_result:
            # Check for anomlies in the results because of too low am and diameter combinations
            # If the velocity is lower than the velocity at sealevel pressure, 
            # set the velocity to the velocity at sealevel pressure
            # Explained in section 4.2.
            sealevel= inputs.copy()
            sealevel[:,1]=101235 #Set pressure to sealevel pressure
            transformed_sealevel = self.transform(sealevel)
            sealevel_velocities= self.calculate(transformed_sealevel)
            # If the velocity is lower than the velocity at sealevel pressure,
            # set the velocity to the velocity at sealevel pressure
            velocities= np.where(velocities<sealevel_velocities,
                                  sealevel_velocities, velocities)
        return np.round(velocities, 2)

# Example usage:
if __name__ == "__main__":
    model = VelocityModel(config="config_model.json")
    print(model.predict([[0.016, 0.3, 2, 0.1],[0.01, 0, 9, 0]]))