
data_dir='./'

/opt/app/neo4j-community-3.3.1/bin/neo4j-admin \
import \
--ignore-duplicate-nodes=true \
--ignore-missing-nodes==true \
--database=medicalKnowledgeDatabase.db \
--id-type string \
\
--nodes:Department ${data_dir}/headers/department_header.csv,\
${data_dir}/data/department.txt \
\
--nodes:Keyword ${data_dir}/headers/label_header.csv,\
${data_dir}/data/label.txt \
\
--relationships:DepartmentLabel ${data_dir}/headers/department_label_header.csv,\
${data_dir}/data/department_label.txt


