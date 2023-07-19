class PDController:
    def __init__(self, Kp, Kd, dt):
        """
        Initialize the PD Controller.

        Args:
            Kp (float): Proportional gain.
            Kd (float): Derivative gain.
            dt (float): Time step size.
        """
        self.Kp = Kp
        self.Kd = Kd
        self.dt = dt
        self.prev_error = 0.0

    def control(self, y, yd):
        """
        Compute the control output based on the current and desired state.

        Args:
            y (float): Current state.
            yd (float): Desired state.

        Returns:
            float: Control output.
        """
        error = yd - y
        d_error = (error - self.prev_error) / self.dt

        u = self.Kp * error + self.Kd * d_error

        self.prev_error = error

        return u


pd = PDController(Kp=2.3, Kd=0.557, dt=0.05)

# Then in your control loop:
# for t in range(1000):
#     y = get_current_state()
#     yd = get_desired_state()
#     control_output = pd.control(y, yd)
#     apply_control(control_output)
