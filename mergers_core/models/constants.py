# Solver constants
# MAX_SOLVER_TIME = 10200  # 2 hours 50 minutes (just under 3 to account for other stuff that might take time)
MAX_SOLVER_TIME = 19800  # 5 hours 30 minutes (just under 6 to account for other stuff that might take time e.g. loading data etc.)
# MAX_SOLVER_TIME = 39600  # 11 hours
NUM_SOLVER_THREADS = 1  # Only use the default SAT search

# To scale variables so they satisfy cp-sat's integer requirements
SCALING = (100000, 5)

MAX_STUDENTS_PER_CAT = 5000
MAX_TOTAL_STUDENTS = 5000

RACE_KEYS = {
    "White": "num_white",
    "Black or African American": "num_black",
    "Hispanic/Latino": "num_hispanic",
    "American Indian or Alaska Native": "num_native",
    "Asian": "num_asian",
    "Native Hawaiian or Other Pacific Islander": "num_pacific_islander",
    "Two or more races": "num_two_or_more",
    "Not Specified": "num_not_specified",
    "Total": "num_total",
}

GRADE_TO_INDEX = {
    "PK": 0,
    "KG": 1,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "7": 8,
    "8": 9,
    "9": 10,
    "10": 11,
    "11": 12,
    "12": 13,
    "13": 14,
}

STATUSES = ["UNKNOWN", "MODEL_INVALID", "FEASIBLE", "INFEASIBLE", "OPTIMAL"]

BLOCKS_FILE = (
    "data/attendance_boundaries/2122/{}/estimated_student_counts_per_block.csv"
)

TRAVEL_TIMES_FILE = "data/travel_times_files/2122/{}/block_to_school_driving_times.json"

STATE = "CA"
DISTRICT_ID = "0602160"

DISSIMILARITY_WEIGHT = 1
POPULATION_CONSISTENCY_WEIGHT = 1
